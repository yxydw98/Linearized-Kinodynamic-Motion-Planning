import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def f(x, y):
    return np.sin(x) + np.cos(y)

def fit_and_plot_plane(ax, x_range, y_range):
    # Generate points within this segment
    x_seg, y_seg = np.meshgrid(np.linspace(x_range[0], x_range[1], 10), np.linspace(y_range[0], y_range[1], 10))
    z_seg = f(x_seg, y_seg)
    
    # Flatten for fitting
    X_seg = np.vstack((x_seg.ravel(), y_seg.ravel())).T
    z_seg = z_seg.ravel()
    
    # Fit a plane (linear regression)
    model = LinearRegression().fit(X_seg, z_seg)
    
    # Predict z values for the meshgrid
    z_pred = model.predict(X_seg).reshape(x_seg.shape)
    
    # Plot the plane
    ax.plot_surface(x_seg, y_seg, z_pred, alpha=0.5, edgecolor='none')

x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
x, y = np.meshgrid(x, y)
z = f(x, y)
# Define the ranges for segments
x_ranges = [(-3, -1), (-1, 1), (1, 3)]
y_ranges = [(-3, -1), (-1, 1), (1, 3)]

# Plot the original function
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis', alpha=0.6)
ax1.set_title('Original Function')

# Plot the approximating planes
ax2 = fig.add_subplot(122, projection='3d')
for x_range in x_ranges:
    for y_range in y_ranges:
        fit_and_plot_plane(ax2, x_range, y_range)
ax2.set_title('Plane-wise Linearization')

plt.show()