import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression



# Step 1: Regenerate sample data with structured nonlinear relationships
num_clusters = 40
np.random.seed(42)  # Ensure reproducibility
num_samples = 200
x_new = np.random.uniform(-10, 10, (num_samples, 2))
y_new = np.random.uniform(-10, 10, (num_samples, 2))

# Example nonlinear combination for z
z_new = np.hstack([
    (np.sin(x_new[:, 0]) + np.cos(y_new[:, 0])).reshape(-1, 1), 
    (np.cos(x_new[:, 1]) + np.sin(y_new[:, 1])).reshape(-1, 1)
])

# Step 2: Apply K-means clustering to the combined x and y
xy_combined = np.hstack((x_new, y_new))
kmeans_new = KMeans(n_clusters=num_clusters, random_state=42).fit(xy_combined)
clusters_new = kmeans_new.labels_

# Fit multi-output linear models within each cluster
multi_cluster_models_new = []
for i in range(num_clusters):
    indices = np.where(clusters_new == i)[0]
    if len(indices) == 0:
        continue
    xy_cluster_new = xy_combined[indices]
    z_cluster_new = z_new[indices]
    model_new = MultiOutputRegressor(LinearRegression()).fit(xy_cluster_new, z_cluster_new)
    multi_cluster_models_new.append((i, model_new))

# Step 3: Evaluate the approach - predict z for the entire dataset and calculate MSE
# predicted_z_new = np.zeros_like(z_new)
# for i, model_new in multi_cluster_models_new:
#     indices = np.where(clusters_new == i)[0]
#     xy_input_new = xy_combined[indices]
#     predicted_z_new[indices] = model_new.predict(xy_input_new)

# Calculate MSE for each dimension of z
np.random.seed(99)  # Different seed for generating new points
num_test_samples = 10
x_test = np.random.uniform(-10, 10, (num_test_samples, 2))
y_test = np.random.uniform(-10, 10, (num_test_samples, 2))

# Generating z based on the same nonlinear combination used for training
z_test = np.hstack([
    (np.sin(x_test[:, 0]) + np.cos(y_test[:, 0])).reshape(-1, 1), 
    (np.cos(x_test[:, 1]) + np.sin(y_test[:, 1])).reshape(-1, 1)
])

# Predict z for the new test data points
xy_test_combined = np.hstack((x_test, y_test))
predicted_z_test = np.zeros_like(z_test)

for i, model_new in multi_cluster_models_new:
    # For each cluster, predict z for the points closest to this cluster's center
    distances = kmeans_new.transform(xy_test_combined)[:, i]  # Distance of each point to the i-th cluster center
    cluster_label_test = np.argmin(kmeans_new.transform(xy_test_combined), axis=1) == i
    if np.any(cluster_label_test):
        xy_input_test = xy_test_combined[cluster_label_test]
        predicted_z_test[cluster_label_test] = model_new.predict(xy_input_test)

# Calculate MSE for the test data
mse_z1_test = mean_squared_error(z_test[:, 0], predicted_z_test[:, 0])
mse_z2_test = mean_squared_error(z_test[:, 1], predicted_z_test[:, 1])

print(mse_z1_test, mse_z2_test)