import pybullet as p
import time
import pybullet_data
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def normalize_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi
# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

timeStep = 1.0/240
p.setTimeStep(timeStep)
# p.setRealTimeSimulation(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=90, cameraPitch=-80, cameraTargetPosition=[0.5, 0, 0.75])


# Load a plane and a table
planeId = p.loadURDF("plane.urdf")
# tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])

# Assuming the table height is around 0.7 meters
table_height = 0.625

# Dimensions for the disks
disk_mass = 1
disk_radius = 0.1
disk_height = 0.02

goal_x_pos = 1
goal_y_pos = 0.25

# Create two disks (as cylinders)
disk1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)
disk2 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)

visGoalID = p.createVisualShape(p.GEOM_CYLINDER,radius=0.1, length=1e-3,rgbaColor=[1.0, 0.0, 0.0, 1.0])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[goal_x_pos, goal_y_pos, 0])

# Position the disks on the table
# Adjust z-coordinate to be table_height plus half of disk height
disk1Id = p.createMultiBody(baseMass=10, baseCollisionShapeIndex=disk1, basePosition=[0.5, 0, disk_height/2])
disk2Id = p.createMultiBody(baseMass=10, baseCollisionShapeIndex=disk2, basePosition=[0.3, 0, disk_height/2])
p.changeDynamics(disk1Id, -1, lateralFriction=0, restitution=1)
p.changeDynamics(disk2Id, -1, lateralFriction=0.8, restitution=1)

# Run the simulation

counter = 0
no_contact_counter = 0

bufferSize = int(input("Training times: "))
no_contact_tolerance = 50

preContact = False
bufferShrinked = False
preControl = 0
prev_pos_diff = (0, 0)
prev_robot_control = (0, 0)
candidate_count = 20
push_timestep = 50
speed_scale = 4
dataset = []

timestep_counter = 0  # Initialize a counter to track the number of timesteps since the last position change

while (len(dataset) < bufferSize):

    # Make the object quasi-static
    # p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0], angularVelocity = [0, 0, 0])
    object_pos, object_ = p.getBasePositionAndOrientation(disk2Id)
    object_ori = p.getEulerFromQuaternion(object_)[0]
    # print("orientation format", object_)
    robot_actual_pos, _ = p.getBasePositionAndOrientation(disk1Id)

    pos_angle = random.uniform(-math.pi, math.pi)
    # pos_angle = math.pi
    oppo_pos_angle = pos_angle + math.pi
    oppo_pos_angle = normalize_angle(oppo_pos_angle)
    # print("pos_angle: ", pos_angle, "oppo_pos_angle", oppo_pos_angle)

    robot_pos_x = object_pos[0] + 0.2 * math.cos(pos_angle)
    robot_pos_y = object_pos[1] + 0.2 * math.sin(pos_angle)

    p.resetBasePositionAndOrientation(disk1Id, (robot_pos_x, robot_pos_y, disk_height/2), (1, 0, 0, 0))
    control_angle = random.uniform(-1.2, 1.2)
    # control_angle = 0.5

    control_x = math.cos(control_angle) * speed_scale
    control_y = math.sin(control_angle) * speed_scale

    for i in range (push_timestep):
        p.resetBaseVelocity(disk1Id, linearVelocity=[control_x, control_y, 0])
        # p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(timeStep)
    p.resetBaseVelocity(disk1Id, linearVelocity=[0, 0, 0])
    current_object_velocity, _ = p.getBaseVelocity(disk2Id)
    # print("current object velocity", current_object_velocity)
    while (math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2 + current_object_velocity[2] ** 2) > 0.0001):
        current_object_velocity, _ = p.getBaseVelocity(disk2Id)
        p.stepSimulation()
        time.sleep(timeStep)
    print("current object velocity", current_object_velocity)
    end_pos, end_ = p.getBasePositionAndOrientation(disk2Id)
    end_ori = p.getEulerFromQuaternion(end_)[0]
    # print("End orientation", end_)
    displacement = math.sqrt((end_pos[0] - object_pos[0]) ** 2 + (end_pos[1] - object_pos[1]) ** 2)
    if (displacement > 0.001):
        object_vel_angle = math.atan2(end_pos[1] - object_pos[1], end_pos[0] - object_pos[0])
        object_ori_change = end_ori - object_ori
        object_vel_angle -= oppo_pos_angle
        object_vel_angle = normalize_angle(object_vel_angle)
        control_angle -= oppo_pos_angle
        control_angle = normalize_angle(control_angle)

        dataset.append((control_angle, object_vel_angle))
        # dataset.append((pos_angle, object_vel_angle))
        # dataset.append((pos_angle, control_angle, object_vel_angle))
        # dataset.append((control_angle, displacement))
        print(len(dataset))

data = np.array(dataset)
# np.save("20_data.npy", data)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-math.pi, math.pi])
ax.set_ylim([-math.pi, math.pi])

# Split the data into separate variables for clarity
object_vel_angles = data[:, 1]
control_angles = data[:, 0]

# Scatter plot
ax.scatter(control_angles, object_vel_angles)

# Label axes
ax.set_xlabel('Control Angle')
# ax.set_ylabel('Position Angle')
ax.set_ylabel('Object Velocity Angle')

plt.show()

# Linearize / Polynomial Regress the collected data
x = data[:, 0]
y = data[:, 1]

linear_model = LinearRegression().fit(x.reshape(-1, 1), y)

degrees = [2, 3, 4, 5]
polynomial_models = []
scores = []

for degree in degrees:
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(x.reshape(-1, 1), y)
    polynomial_models.append(poly_model)
    
    # Predict and evaluate
    y_pred = poly_model.predict(x.reshape(-1, 1))
    score = r2_score(y, y_pred)
    scores.append(score)

# Score of the linear model
linear_score = r2_score(y, linear_model.predict(x.reshape(-1, 1)))

# Combine all scores for comparison
all_scores = [linear_score] + scores
models = ['Linear'] + [f'Poly {degree}' for degree in degrees]

# Display the scores
print(list(zip(models, all_scores)))

# best_model = linear_model
best_model = polynomial_models[3]  # Index 1 corresponds to degree 3

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.title('Best Fit: Polynomial Regression (Degree 3)')
plt.xlabel('X')
plt.ylabel('Y')

x_fit = np.linspace(x.min(), x.max(), 400)
y_fit = best_model.predict(x_fit.reshape(-1, 1))
plt.plot(x_fit, y_fit, color='red', label='Polynomial Fit (Degree 3)')
plt.legend()
plt.show()

p.resetBasePositionAndOrientation(disk1Id, (0.5, 0, disk_height/2), (1, 0, 0, 0))
p.resetBasePositionAndOrientation(disk2Id, (0.3, 0.1, disk_height/2), (1, 0, 0, 0))

counter = 0

# while True:
#     counter += 1
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
#     colors = ['r', 'g', 'b', 'y', 'c']
    
#     p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
#     object_pos, _ = p.getBasePositionAndOrientation(disk2Id)
#     robot_pos, _ = p.getBasePositionAndOrientation(disk1Id)
#     pos_angle = math.atan2((robot_pos[1] - object_pos[1]), (robot_pos[0] - object_pos[0]))
#     oppo_pos_angle = pos_angle + math.pi
#     oppo_pos_angle = oppo_pos_angle - 2 * math.pi if oppo_pos_angle > math.pi else oppo_pos_angle
#     print("pos_angle: ", pos_angle, "oppo_pos_angle", oppo_pos_angle)

#     desired_object_vel_angle = math.atan2(goal_y_pos - object_pos[1], goal_x_pos - object_pos[0])
#     desired_object_vel_angle -= oppo_pos_angle
#     desired_object_vel_angle = normalize_angle(desired_object_vel_angle)

#     start_time = time.time()

#     x_range = np.linspace(-np.pi/2, np.pi/2, 1000)
#     y_pred = best_model.predict(x_range.reshape(-1, 1))

#     differences = np.abs(y_pred - desired_object_vel_angle)

#     min_diff_x = np.argmin(differences)
#     x_closest = x_range[min_diff_x]

#     end_time = time.time()
#     time_used = end_time - start_time
#     print("time_used, ", time_used)
#     x_closest += oppo_pos_angle

#     control_x = math.cos(x_closest)
#     control_y = math.sin(x_closest)
#     for i in range (push_timestep):
#         p.resetBaseVelocity(disk1Id, linearVelocity=[control_x, control_y, 0])
#         p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
#         p.stepSimulation()
#         time.sleep(timeStep)
