import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load the dataset
data_path = 'cube_push_1000.npy'
data = np.load(data_path)



# Check the shape and some basic information about the data to understand its structure
data.shape, data[:5]

# Split the dataset into inputs (X) and output (y)


for element in data:
    if (element[2] < -math.pi / 2 and element[0] > math.pi / 2):
        element[2] += math.pi * 2
    if (element[2] > math.pi / 2 and element[0] < -math.pi / 2):
        element[2] -= math.pi * 2
    if (element[1] < math.pi / 2 and element[0] > math.pi / 2):
        element[1] += math.pi * 2
    if (element[1] > math.pi * 1.5 and element[2] < -math.pi):
        element[1] -= math.pi * 2

delete_mask = (data[:, 1] > 5) & (data[:, 0] < -2)
keep_mask = ~delete_mask
data = data[keep_mask]

np.save("concatenated_cube_data.npy", data)
X = data[:, :2]  # all rows, first two columns
y = data[:, 2]   # all rows, third column
# new_column = data[:, 1] - data[:, 2]
# new_column = new_column.reshape(-1, 1)
# data_with_new_column = np.hstack((data, new_column))



# Visualizing the dataset in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o')
ax.set_xlabel('object_velocity_angle')
ax.set_ylabel('pos_angle')
ax.set_zlabel('control_angle')
ax.set_title('3D Scatter plot of the Dataset')

plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predictions for visualization
y_pred_lin = lin_reg.predict(X)

# Calculate metrics for Linear Regression
mse_lin = mean_squared_error(y, y_pred_lin)
r2_lin = r2_score(y, y_pred_lin)

# Polynomial Features Transformation
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X)

# Polynomial Regression
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predictions for visualization
y_pred_poly = poly_reg.predict(X_poly)

# Calculate metrics for Polynomial Regression
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)


x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
                             np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
X_surf = np.c_[x_surf.ravel(), y_surf.ravel()]

# Predictions on the meshgrid for Linear Regression
y_pred_surf_lin = lin_reg.predict(X_surf).reshape(x_surf.shape)

# Predictions on the meshgrid for Polynomial Regression
X_surf_poly = poly_features.transform(X_surf)
y_pred_surf_poly = poly_reg.predict(X_surf_poly).reshape(x_surf.shape)

# Plotting
fig = plt.figure(figsize=(20, 10))

# Linear Regression
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Actual Data')
ax1.plot_surface(x_surf, y_surf, y_pred_surf_lin, color='r', alpha=0.3)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Output')
ax1.set_title('Linear Regression Fit')

# Polynomial Regression
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Actual Data')
ax2.plot_surface(x_surf, y_surf, y_pred_surf_poly, color='g', alpha=0.3)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Output')
ax2.set_title('Polynomial Regression Fit')

plt.show()