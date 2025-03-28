import numpy as np # type: ignore
from sympy import symbols, sqrt, atan2, cos, sin, Matrix, lambdify, eye, asin # type: ignore
from jacobians import * # type: ignore

def tendon2coordinates(DELTAL0, DxDyDl0, coords0, d_DELTAL, step_size=1):
    """
    Compute the new configuration from tendon length variations with adaptive discretization.

    Output:
    - DELTAL   <- tendon lengths of the new configuration
    - DxDyDl   <- curvature of the new configuration
    - coords   <- (x,y,z,yaw,pitch,roll) of the new configuration

    Input:
    - DELTAL0  <- tendon lengths of the old configuration
    - DxDyDl0  <- curvature of the old configuration
    - coords0  <- (x,y,z,yaw,pitch,roll) of the old configuration
    - d_DELTAL <- tendon length variations (DELTAL - DELTAL0)
    - step_size <- step granularity for discretization (default: 2)
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.array([1e-5, 1e-5, 1e-5])

    # Compute the number of steps based on d_DELTAL magnitude
    max_change = np.linalg.norm(d_DELTAL)
    N = max(int(np.ceil(max_change / step_size)), 5)  # Ensure at least 5 steps

    # Compute step-wise variations
    d_DELTAL_step = d_DELTAL / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    coords = np.copy(coords0)
    
    Q2TENDON = q2tendon()
    Q2COORDINATES = q2coordinates()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_tendon = Q2TENDON(*DxDyDl)

        # Compute the incremental variation in curvature
        d_DxDyDl = np.linalg.pinv(J_tendon) @ d_DELTAL_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl

        # Compute the incremental variation in coordinates
        J_coords = Q2COORDINATES(*DxDyDl)
        d_coords = J_coords @ d_DxDyDl

        # Update coordinates and tendon lengths
        coords = coords + d_coords
        DELTAL = DELTAL + d_DELTAL_step

    return DELTAL, DxDyDl, coords

def coordinates2tendon(DELTAL0, DxDyDl0, coords0, d_xyz, step_size=1, control=False):
    """
    Compute the new tendon lengths from coordinate variations with adaptive discretization.

    Output:
    - DELTAL   <- tendon lengths of the new configuration
    - DxDyDl   <- curvature of the new configuration
    - coords   <- (x,y,z,yaw,pitch,roll) of the new configuration

    Input:
    - DELTAL0  <- tendon lengths of the old configuration
    - DxDyDl0  <- curvature of the old configuration
    - coords0  <- (x,y,z,yaw,pitch,roll) of the old configuration
    - d_xyz    <- coordinate variations (in x,y,z)
    - step_size <- step granularity for discretization
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.array([1e-5, 1e-5, 1e-5])

    # Compute the number of steps based on d_xyz magnitude
    max_change = np.linalg.norm(d_xyz)
    N = max(int(np.ceil(max_change / step_size)), 5)  # Ensure at least 5 steps

    # Compute step-wise variations
    d_xyz_step = d_xyz / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    coords = np.copy(coords0)
    
    Q2COORDINATES = q2coordinates()
    Q2TENDON = q2tendon()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_coords = Q2COORDINATES(*DxDyDl)
        J_xyz = J_coords[0:3, 0:3]

        # Compute incremental variation in curvature
        d_DxDyDl = np.linalg.pinv(J_xyz) @ d_xyz_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl

        # Compute the incremental variation in coordinates
        d_coords = J_coords @ d_DxDyDl
        coords = coords + d_coords

        # Compute tendon Jacobian and incremental tendon variation
        J_tendon = Q2TENDON(*DxDyDl)
        d_DELTAL = J_tendon @ d_DxDyDl

        # Update tendon lengths
        DELTAL = DELTAL + d_DELTAL

        # Check motion boundaries
        D = np.sqrt(DxDyDl[0] ** 2 + DxDyDl[1] ** 2)
        Dl_min = -30    # measures 
        D_max = 35     # measures (def.s)

        if (DxDyDl[2] > Dl_min or D > D_max) and control:
            return DELTAL0, DxDyDl0, coords0  # boundaries

    return DELTAL, DxDyDl, coords



