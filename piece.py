import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator

# Create a 2D dataset
x_2d, y_2d = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
z_2d = x_2d * np.sin(y_2d)

# Flatten the 2D arrays
x_flat = x_2d.flatten()
y_flat = y_2d.flatten()
z_flat = z_2d.flatten()

print("x", x_2d)
print(x_2d.shape)
print("x_flat", x_flat)
print(x_flat.shape)

# Piecewise linear interpolation
interpolator = LinearNDInterpolator(list(zip(x_flat, y_flat)), z_flat)

# Define a grid for visualization
x_vis, y_vis = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
z_vis = interpolator(x_vis, y_vis)
print("x_vis shape", x_vis.shape)
print("z_vis shape", z_vis.shape)

# Plotting
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_flat, y_flat, z_flat, color='r', label='Original Data Points')
ax.plot_surface(x_vis, y_vis, z_vis, cmap='viridis', edgecolor='none', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Piecewise Linear Interpolation Over a 2D Dataset')
ax.legend()
plt.show()

# Compare interpolated values with actual values for new points
x_compare = np.array([1, 3])
y_compare = np.array([2, 4])
z_actual = x_compare * np.sin(y_compare)
z_predicted = interpolator(x_compare, y_compare)
comparison = np.vstack((z_actual, z_predicted, np.abs(z_actual - z_predicted))).T

# Create a comparison dataframe
comparison_df = pd.DataFrame(comparison, columns=['Z Actual', 'Z Predicted', 'Absolute Error'])
comparison_df.index = ['Point {}'.format(i + 1) for i in range(len(x_compare))]
print(comparison_df)