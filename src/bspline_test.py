import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 显式导入 3D 绘图工具箱，防止老版本 matplotlib 报 'Unknown projection 3d' 错误
from mpl_toolkits.mplot3d import Axes3D 

def generate_bspline():
    print("Generating B-Spline trajectory...")

    # ==========================================
    # 1. Define 3D waypoints for the robot arm
    # A smooth "U" shaped trajectory above the table
    # ==========================================
    waypoints = np.array([
        [0.3, -0.2, 0.1],  # Start point (x, y, z) in meters
        [0.4, -0.2, 0.1],  # Pre-turn
        [0.5,  0.0, 0.1],  # Curve apex
        [0.4,  0.2, 0.1],  # Post-turn
        [0.3,  0.2, 0.1]   # End point
    ])
    
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    # ==========================================
    # 2. Generate B-Spline Curve
    # ==========================================
    # splprep: parametric B-spline curve representation
    # k=3: Cubic B-spline (C2 continuous, smooth acceleration)
    # s=0: Force the curve to pass through all waypoints (interpolation)
    tck, u = splprep([x, y, z], s=0, k=3)

    # Generate 500 uniformly spaced u parameters from 0 to 1
    u_fine = np.linspace(0, 1, 500)

    # splev: Evaluate the B-spline curve at given u points
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    # ==========================================
    # 3. Calculate actual physical distance between points
    # This represents the true physical speed profile
    # ==========================================
    distances = []
    for i in range(1, len(x_fine)):
        dx = x_fine[i] - x_fine[i-1]
        dy = y_fine[i] - y_fine[i-1]
        dz = z_fine[i] - z_fine[i-1]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(dist)

    # ==========================================
    # 4. Visualization
    # ==========================================
    fig = plt.figure(figsize=(12, 5))

    # ---- Plot 1: 3D Trajectory Space ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_fine, y_fine, z_fine, 'b-', linewidth=2, label='B-Spline Curve (k=3)')
    ax1.scatter(x, y, z, color='red', s=50, label='Control Points (Waypoints)')
    
    ax1.set_title("3D Trajectory Planning for Robot End-Effector")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    # ---- Plot 2: Speed Profile Analysis ----
    ax2 = fig.add_subplot(122)
    # Plot the physical step size (representing velocity)
    ax2.plot(u_fine[1:], distances, 'g-', linewidth=2)
    ax2.fill_between(u_fine[1:], distances, color='green', alpha=0.2)
    
    ax2.set_title("Actual Physical Speed when Parameter u Increases Uniformly")
    ax2.set_xlabel("Curve Parameter u (0 -> 1)")
    ax2.set_ylabel("Physical Step Size / Velocity Profile (m/step)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_bspline()