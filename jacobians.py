import numpy as np
from sympy import symbols, sqrt, atan2, cos, sin, Matrix, lambdify, eye, asin

# Units of measure:
# - LENGTHS: [mm]
# - ANGLES: [rad]

# q2tendon(): from q to DELTAL
def q2tendon():
    """Jacobian of tendons ([Dx,Dy,Dl] to [DELTAL])"""

    # Components
    Dx, Dy, Dl = symbols("Dx Dy Dl")
    diameter = 60      # tuned base diameter [mm]
    d = diameter/2     # constant section

    # Changes of variables to phi and theta (Dl = Dl)
    D = sqrt(Dx**2 + Dy**2 + 1e-8)  # curvature
    theta = D/d                     # theta
    phi = atan2(Dy,Dx + 1e-8)

    # Positions of tendons (assuming origin is coherent here)
    delta = Matrix([0, np.radians(120), np.radians(-120)])

    # Variations of lengths [deltaL1; deltaL2; deltaL3]
    DELTAL = Matrix([Dl - theta * d * cos(delta_i - phi) for delta_i in delta])

    # Define variables for computation
    variables_vector = Matrix([Dx, Dy, Dl])

    # Jacobian
    J = DELTAL.jacobian(variables_vector)

    return lambdify((Dx, Dy, Dl), J, modules=['numpy'])

# q2coordinates(): from q to the CNN-FCN output
def q2coordinates():
    """Jacobian of the helyx ([Dx,Dy,Dl] to [x,y,z,yaw,pitch,roll])"""

    # Components
    Dx, Dy, Dl = symbols("Dx Dy Dl")
    diameter = 60     # tuned base diameter [mm]
    d = diameter / 2  # constant section
    L0 = 140          # rest length[mm]

    # Computation of curvature
    D = sqrt(Dx ** 2 + Dy ** 2 + 1e-8)  # curvature

    # Robot kinematics (from paper):
    # - Rotation matrix
    R = Matrix([
                  [1+Dx**2/D**2*(cos(D/d)-1), Dx*Dy/D**2*(cos(D/d)-1), -Dx/D*sin(D/d)],
                  [Dx*Dy/D**2*(cos(D/d)-1), 1+Dy**2/D**2*(cos(D/d)-1), -Dy/D*sin(D/d)],
                  [Dx/D*sin(D/d), Dy/D*sin(D/d), cos(D/d)]
               ])
    # - Translation vector (center of the section)
    t = Matrix([
                [(d*(L0+Dl)/D**2)*Dx*(1-cos(D/d))],
                [(d*(L0+Dl)/D**2)*Dy*(1-cos(D/d))],
                [(d*(L0+Dl)/D**2)*D*sin(D/d)]
               ])

    # Convert rotation matrix to Yaw-Pitch-Roll (ZYX, intrinsic)
    yaw = atan2(R[1, 0], R[0, 0] + 1e-8)    # LaValle confirms
    pitch = asin(- R[2, 0])                 # LaValle confirms
    roll = atan2(R[2, 1], R[2, 2] + 1e-8)   # LaValle confirms
    
    # Vector with [translations; quaternions]
    T = Matrix.vstack(t, Matrix([yaw, pitch, roll]))

    # Define variables for computation
    variables_vector = Matrix([Dx, Dy, Dl])

    # Jacobian
    J = T.jacobian(variables_vector)

    return lambdify((Dx, Dy, Dl), J, modules=['numpy'])
