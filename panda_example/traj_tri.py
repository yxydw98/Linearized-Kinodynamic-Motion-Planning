import numpy as np
import matplotlib.pyplot as plt

trajectory_data = np.load('triangle_traj.npy')

# Plot the trajectory in 3D space
plt.figure()
plt.plot(trajectory_data[:, 0], trajectory_data[:, 1], label='2D Trajectory')
plt.axis('equal')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('2D Trajectory Plot')
plt.show()