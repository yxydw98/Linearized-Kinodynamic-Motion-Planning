import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# Load the dataset
data_path = 'cylinder_push_1000.npy'  # Update this path
dataset = np.load(data_path)

object_vel_angles = dataset[:, 0]
pos_angles = dataset[:, 1]
control_angles = dataset[:, 2]
<<<<<<< HEAD
adjusted_control_angles = np.where(control_angles > np.pi, control_angles - 2 * np.pi, control_angles)
dataset[:, 2] = adjusted_control_angles
np.save('adjusted_data_cylinder_push_1000.npy', dataset)
=======


>>>>>>> 8ece701567e2f68c9f37ba437ed1ecb514d50c8a
sin_object_vel_angles = np.sin(object_vel_angles)
sin_pos_angles = np.sin(pos_angles)
sin_control_angles = np.sin(control_angles)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
<<<<<<< HEAD
ax.scatter(object_vel_angles, pos_angles, adjusted_control_angles)
=======
ax.scatter(object_vel_angles, pos_angles, control_angles)
>>>>>>> 8ece701567e2f68c9f37ba437ed1ecb514d50c8a

# ax.plot(object_vel_angles, pos_angles, sin_control_angles, label='Sine of Control Angles', color='r')
# ax.plot(object_vel_angles, sin_pos_angles, control_angles, label='Sine of Position Angles', color='g')
# ax.plot(sin_object_vel_angles, pos_angles, control_angles, label='Sine of Object Velocity Angles', color='b')


ax.set_xlabel('Object Velocity Angle')
ax.set_ylabel('Position Angle')
ax.set_zlabel('Control Angle')
ax.legend()
plt.show()

# Splitting dataset into features (X) and target (y)
X = dataset[:, 1:]  # Relative position angle and control angle
y = dataset[:, 0]  # Resultant second cylinder velocity angle

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training a Random Forest Regressor
start_time = time.time()
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions with the Random Forest model
y_pred_rf = rf_model.predict(X_test)

end_time = time.time()

# Calculating performance metrics for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Displaying the performance metrics
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"R-squared (R²): {r2_rf}")

# Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Predictions")
plt.plot(y_test, y_test, color="red", label="Perfect Fit")  # Line for perfect predictions
plt.title("Random Forest Model: Actual vs. Predicted Values")
print(f"Training time: {end_time - start_time} seconds")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)
plt.show()