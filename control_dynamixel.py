# Libraries
import numpy as np # type: ignore
import sys
import time
import rospy
import os
from datetime import datetime
from dynamixel_sdk import *
from geometry_msgs.msg import PoseStamped
import cv2 # type: ignore
from scipy.spatial.transform import Rotation as R # type: ignore
import threading
from ctypes import c_uint32
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import termios
import tty

# User-defined
from jacobians import *
from kinematics import *

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# ======================================================================================
#                                    Functions
# ======================================================================================

# Write register
def write_register(motor_id, address, value, size=1):
    """ Writes a single register """
    if size == 1:
        packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
    elif size == 2:
        packet_handler.write2ByteTxRx(port_handler, motor_id, address, value)
    elif size == 4:
        packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)

def get_key():
    """Read a single character without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Pose callback for positions and quaternions
def pose_callback(msg, topic_name):
    """Callback function to process messages from each topic"""
    global positions, quaternions, angles
    
    # Extract position and orientation
    position = msg.pose.position
    orientation = msg.pose.orientation

    # Update stored data
    positions[topic_name] = [position.x, position.y, position.z]
    quaternions[topic_name] = [orientation.x, orientation.y, orientation.z, orientation.w]  # scalar last

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    angles[topic_name] = R.from_quat(quaternions[topic_name]).as_euler('zyx')

# Move motors
def move_motors(d_DELTAL, TOLERANCE=20):
    """
    Moves motors by d_DELTAL increment and returns the actual delta achieved.
    Encoder values are read as uint32 and explicitly reinterpreted as int32 (Dynamixel standard).
    All computations are in float32 to avoid overflow.
    """

    # Read position as signed int32 (from raw uint32) and convert to float
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    # Initial positions
    start_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Target positions
    goal_positions = start_positions + (unit_scale * d_DELTAL).astype(np.float32)

    # Send goal to motors
    groupSyncWrite = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)
    for motor_id, pos in zip(MOTOR_IDs, goal_positions):
        pos_int = int(round(pos))  # Cast to int for sync write
        param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                 DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
        groupSyncWrite.addParam(motor_id, param)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    # Wait for completion or timeout
    tic = time.time()
    while time.time() - tic < 1.5:
        current = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
        if np.all(np.abs(current - goal_positions) <= TOLERANCE):
            break
        time.sleep(0.05)

    # Final positions
    end_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Compute delta in mm
    d_tick = end_positions - start_positions
    d_DELTAL_real = d_tick / unit_scale

    return d_DELTAL_real

# Calibration function
def calibrate_motors():
    """
    Interactive calibration for each motor.
    Press:
      - 'w' to move forward (with adaptive increment)
      - 's' to move backward (with adaptive increment)
      - 'h' to confirm and go to the next motor
    """
    print("--- Calibration started ---")
    base_step = np.pi * D_pulley / 12  # [mm]
    max_multiplier = 10
    accel_threshold = 0.4  # seconds between presses for acceleration

    for idx, motor_id in enumerate(MOTOR_IDs):
        print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
        last_key = None
        last_time = time.time()
        step_multiplier = 1

        while True:
            key = get_key()
            now = time.time()

            # Reset multiplier if too slow or switched key
            if key != last_key or (now - last_time) > accel_threshold:
                step_multiplier = 1
            else:
                step_multiplier = min(step_multiplier + 1, max_multiplier)

            last_time = now
            last_key = key

            if key == 'h':
                break
            elif key in ['w', 's']:
                d_DEL = np.zeros(MOTOR_Num, dtype=np.float32)
                direction = 1 if key == 'w' else -1
                step = direction * base_step * step_multiplier
                d_DEL[idx] = step
                move_motors(d_DEL)
            else:
                print("\n Invalid input. Use 'h', 'w', or 's'.")

    print("\n --- Calibration completed ---")

# Relative coordinates function
def relative_coordinates(x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
                         x_base, y_base, z_base, yaw_base, pitch_base, roll_base):
    """
    Compute the position and orientation of the tip relative to the base,
    both given in mocap global frame. Output is in the robot frame.
    """

    # Mapping from mocap to robot frame:
    # x -> x, y -> -z, z -> y
    R_corr = np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0]
    ])

    # Absolute positions
    p_tip_global = np.array([x_tip, y_tip, z_tip])
    p_base_global = np.array([x_base, y_base, z_base])

    # Position difference in mocap frame
    p_rel_global = p_tip_global - p_base_global

    # Apply mocap → robot frame
    p_rel_robot = R_corr @ p_rel_global

    # Rotation matrices from mocap (ZYX)
    R_tip_global = R.from_euler('zyx', [yaw_tip, pitch_tip, roll_tip]).as_matrix()
    R_base_global = R.from_euler('zyx', [yaw_base, pitch_base, roll_base]).as_matrix()

    # Rotation from base to tip
    R_rel = R_base_global.T @ R_tip_global

    # Convert to ZYX Euler angles
    euler_rel = R.from_matrix(R_rel).as_euler('zyx')

    return np.concatenate((p_rel_robot * 1000, euler_rel))  # mm + radians

# ======================================================================================
#                                   Dynamixel Setting
# ======================================================================================

DEVICENAME      = '/dev/ttyUSB0'
BAUDRATE        = 57600
PROTOCOL_VERSION = 2.0
EXPOSITION_MODE = 4			# position + current (to be changed)

MOTOR_IDs = np.array([1, 2, 3])	# 3 motors (as in the setup)
MOTOR_Num = MOTOR_IDs.shape[0]	# number of motors

D_pulley = 6				# [mm]
unit_scale = 4096/(np.pi*D_pulley)	# motor position/DELTAL

# ======================================================================================
#                                   Dynamixel Setup
# ======================================================================================

# Initialize PortHandler
port_handler = PortHandler(DEVICENAME)

# Initialize PacketHandler
packet_handler = Protocol2PacketHandler()

# Open port
if not port_handler.openPort():
    print("Failed to open port")
    sys.exit(1)
print("Port opened successfully")

# Set baudrate
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    sys.exit(1)
print("Baudrate set successfully")

# Define Dynamixel control table addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_PROFILE_VELOCITY = 112

# Set operating mode for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_OPERATING_MODE, EXPOSITION_MODE, size=1)

# Enable torque for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_TORQUE_ENABLE, 1, size=1)

VELOCITY_LIMIT = 100     # velocity limitation

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_PROFILE_VELOCITY, VELOCITY_LIMIT, size=4)


print("--- Reference positions set to zero ---")

# ======================================================================================
#                                Calibrate The Motors
# ======================================================================================

calibrate_motors()

# ======================================================================================
#                                   Motion Capture Setup
# ======================================================================================

# Initialize ROS node
rospy.init_node('optitrack_listener')

# Dictionaries to store positions and angles informations
positions = {}
quaternions = {}
angles = {}

# Define the TOPICS to subscribe to:
# - Base (static): BASE
# - Tip (dynamic): TIP
# Define the TOPICS to subscribe to:
helyx_topics = [
    ('/optitrack/BASE/pose', 'BASE'),
    ('/optitrack/TIP/pose', 'TIP')
]

# Create subscribers for the topics
subscribers = []
for topic_name, key in helyx_topics:
    sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)   # use pose_callback
    subscribers.append(sub)

# Allow messages to start arriving
time.sleep(1)
    
# ======================================================================================
#                           Get to Wanted Configuration (e.g.)
# ======================================================================================

# Initial conditions (rest position)
DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
coords0 = np.array([1e-4, 1e-4, 140, 1e-4, 1e-4, 1e-4])

# Decide if it is needed to try (default no)
tryFlag = False

if tryFlag:
    # Ask user for coordinate variations (deltax, deltay, deltaz)
    dx = float(input("Enter deltax [mm]: "))
    dy = float(input("Enter deltay [mm]: "))
    dz = float(input("Enter deltaz [mm]: "))
    d_xyz = np.array([dx, dy, dz])

    # Call coordinates2tendon to compute new DELTAL
    DELTAL, DxDyDl, coords = coordinates2tendon(DELTAL0, DxDyDl0, coords0, d_xyz)

    # Print results
    print("\nComputed DELTAL values:")
    print(DELTAL)

    # Application of move_motors()
    move_motors(DELTAL - DELTAL0)
    time.sleep(1)

# ======================================================================================
#                                   Camera Reading
# ======================================================================================

# Use v4l2-ctl --list-devices for the usb camera
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

def capture_image():
    """Captures an image from the USB camera and returns it as a NumPy array, or None if the capture fails."""
    
    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        return None

    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Failed to capture image")
        return None

    return frame

# Capture and show image
image = capture_image()
if image is not None:
    cv2.imshow("Camera Output", image)
    cv2.waitKey(1000)   # ms
    cv2.destroyAllWindows()
else:
    print("[ERROR] No image captured.")

time.sleep(3)

# ======================================================================================
#                               Samples Cycles
# ======================================================================================

# Calibrate motors
print("--- Press 'y' to calibrate the motors, ENTER to go on ---")
key = get_key()
if key == 'y':
  calibrate_motors()
  print("\n")

# Initial conditions (rest position, repeated)
DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
coords0 = np.array([1e-4, 1e-4, 140, 1e-4, 1e-4, 1e-4])

# Sampling movements (datasets)
y = []			    # ground truth
X_images = []		# images

y_dDELTAL = []      # dDELTAL
X_xyz0 = []         # xyz0
X_dxyz = []         # dxyz
X_DELTAL0 = []      # DELTAL0

diameter = 60  # tuned base diameter [mm]
d = diameter / 2  # constant section
L0 = 140  # rest length [mm]

# Print initial position
x_abs, y_abs, z_abs = positions.get('TIP')
roll_abs, pitch_abs, yaw_abs = angles.get('TIP')
x_base, y_base, z_base = positions.get('BASE')
roll_base, pitch_base, yaw_base = angles.get('BASE')
mocap_xyzZYX = relative_coordinates(x_abs,y_abs,z_abs,roll_abs,pitch_abs,yaw_abs,x_base,y_base,z_base,roll_base,pitch_base,yaw_base)
xyz1 = mocap_xyzZYX

print(f"Initial position = {mocap_xyzZYX}")

# Define initial position
DELTAL1, _, _ = coordinates2tendon(DELTAL0, DxDyDl0, coords0, mocap_xyzZYX[:3] - coords0[:3])

# Mocap data for safety
mocap_BASE = []
mocap_TIP = []

time.sleep(3)

N_data = 1000
for i in range(N_data):

    # Sample d_DELTAL
    DELTAL_sample = np.random.uniform(low=-65, high=30, size=3)
    X_DELTAL0.append(DELTAL1)

    # Move and save position
    d_DELTAL_real = move_motors(DELTAL_sample - DELTAL1)
    DELTAL1 = DELTAL1 + d_DELTAL_real

    # Print step
    print(f"Datum {i+1}")
    print(f"DELTAL = {DELTAL1}")

    # Recording sensor data from the movement (ZYX)
    x_abs, y_abs, z_abs = positions.get('TIP')
    yaw_abs, pitch_abs, roll_abs = angles.get('TIP')
    x_base, y_base, z_base = positions.get('BASE')
    yaw_base, pitch_base, roll_base = angles.get('BASE')
    mocap_xyzZYX = relative_coordinates(x_abs,y_abs,z_abs,yaw_abs,pitch_abs,roll_abs,x_base,y_base,z_base,yaw_base,pitch_base,roll_base)
    mocap_TIP.append([x_abs,y_abs,z_abs,yaw_abs,pitch_abs,roll_abs])
    mocap_BASE.append([x_base,y_base,z_base,yaw_base,pitch_base,roll_base])

    # For the dataset
    image_dataset = capture_image()

    # Capture and show image
    if image_dataset is not None:
        cv2.imshow("Camera Output", image_dataset)
        cv2.waitKey(1)                              # 1 ms
        image_rgb = cv2.cvtColor(image_dataset, cv2.COLOR_BGR2RGB)
    else:
        print("[ERROR] No image captured.")

    # Construction of datasets
    y.append(mocap_xyzZYX)
    X_images.append(image_rgb)

    # For the Jacobian
    y_dDELTAL.append(d_DELTAL_real)
    X_xyz0.append(xyz1)
    X_dxyz.append(mocap_xyzZYX - xyz1)

    # Update of xyz1
    xyz1 = mocap_xyzZYX

    time.sleep(0.1)

# Conversion for saving
mocap_BASE = np.array(mocap_BASE, dtype=np.float32)
mocap_TIP = np.array(mocap_TIP, dtype=np.float32)
y = np.array(y, dtype=np.float32)
X_images = np.array(X_images, dtype=np.uint8)

# Jacobian
y_dDELTAL = np.array(y_dDELTAL, dtype=np.float32)   # note that for the final one it is enough to apply cumsum (with DELTAL1 found in the same way): inverse kinematics
X_xyz0 = np.array(X_xyz0, dtype = np.float32)       # first MOCAP value
X_dxyz = np.array(X_dxyz, dtype = np.float32)
X_DELTAL0 = np.array(X_DELTAL0, dtype = np.float32)

# Save datasets for Google Colab/TensorFlow usage
np.savez('Dataset12.npz', y=y, X_images=X_images)
np.savez('Backup_mocap12.npz', mocap_BASE=mocap_BASE, mocap_TIP=mocap_TIP)
np.savez('Jacobian12.npz', y_dDELTAL=y_dDELTAL, X_dxyz=X_dxyz, X_xyz0=X_xyz0, X_DELTAL0=X_DELTAL0)

print(f"--- Data collection completed and saved, for {N_data} points ---")

# =======================================================================================
#                              Close the Camera
# =======================================================================================

# Closing
cap.release()
cv2.destroyAllWindows()