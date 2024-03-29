import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import lstsq
from sklearn.cluster import KMeans

# Load the dataset
data = np.load('concatenated_data.npy')

# Apply KMeans clustering to identify three clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Corrected function to fit a plane to a set of points
def fit_plane_corrected(points):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    A = np.c_[X, Y, np.ones(X.shape[0])]
    coeff, _, _, _ = lstsq(A, Z)
    return coeff

# Fit a plane for each cluster using the corrected approach
planes_corrected = []
for i in range(3):
    cluster_points = data[clusters == i]
    coeff = fit_plane_corrected(cluster_points)
    planes_corrected.append(coeff)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a different color
colors = ['r', 'g', 'b']
for i in range(3):
    cluster_points = data[clusters == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[i], marker='o')

# Function to plot planes
def plot_plane(ax, coeffs, color):
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=10)
    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=10)
    X, Y = np.meshgrid(x, y)
    Z = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    ax.plot_surface(X, Y, Z, alpha=0.3, color=color)

# Plot each plane
for i, coeffs in enumerate(planes_corrected):
    plot_plane(ax, coeffs, colors[i])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D Scatter Plot of Data with Fitted Planes')
plt.show()
