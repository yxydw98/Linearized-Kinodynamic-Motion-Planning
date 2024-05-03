import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.spatial.distance import cdist

# Load the trajectory data
def find_closest_points(actual_trajectory, desired_trajectory):
    distances = cdist(desired_trajectory, actual_trajectory)
    closest_indices = np.argmin(distances, axis=1)
    closest_points = actual_trajectory[closest_indices]

    return closest_points, closest_indices

trajectory_cylinder_data = np.load('cylinder_traj.npy')
trajectory_cube_data = np.load('cube_traj.npy')
trajectory_triangle_data = np.load('triangle_traj.npy')

desired_trajectory = utils.generate_circular_trajectory(origin=[0.0, 0.0], radius=0.1)

closest_points, closest_indices = find_closest_points(trajectory_cube_data[:, :2], desired_trajectory)
print("Closest points indices in the actual trajectory", closest_indices)

mae = np.mean(np.abs(desired_trajectory - closest_points))
print("MAE", mae)
# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 3 subplots in a row, with a total width of 15 and height of 5

# Plot cylinder trajectory
axs[0].plot(trajectory_cylinder_data[:, 0], trajectory_cylinder_data[:, 1], label='Cylinder Trajectory')
axs[0].set_title('Cylinder Trajectory')
axs[0].set_xlabel('X Position')
axs[0].set_ylabel('Y Position')
axs[0].axis('equal')
axs[0].legend()

# Plot cube trajectory
axs[1].plot(trajectory_cube_data[:, 0], trajectory_cube_data[:, 1], label='Cube Trajectory')
axs[1].set_title('Cube Trajectory')
axs[1].set_xlabel('X Position')
axs[1].set_ylabel('Y Position')
axs[1].axis('equal')
axs[1].legend()

# Plot triangle trajectory
axs[2].plot(trajectory_triangle_data[:, 0], trajectory_triangle_data[:, 1], label='Triangle Trajectory')
axs[2].set_title('Triangle Trajectory')
axs[2].set_xlabel('X Position')
axs[2].set_ylabel('Y Position')
axs[2].axis('equal')
axs[2].legend()

# Show the plot
plt.tight_layout()  # Adjust layout to not overlap
plt.savefig('circular_trajectories_plot.pdf')
plt.show()