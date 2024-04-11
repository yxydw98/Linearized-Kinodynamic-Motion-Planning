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


# def 
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

goal_x_pos = -0.5
goal_y_pos = 0.25

# Create two disks (as cylinders)
disk1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)
disk2 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)

visGoalID = p.createVisualShape(p.GEOM_CYLINDER,radius=0.1, length=1e-3,rgbaColor=[1.0, 0.0, 0.0, 1.0])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[goal_x_pos, goal_y_pos, 0])

# Position the disks on the table
# Adjust z-coordinate to be table_height plus half of disk height
disk1Id = p.createMultiBody(baseMass=100, baseCollisionShapeIndex=disk1, basePosition=[0.5, 0, disk_height/2])
disk2Id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=disk2, basePosition=[0.3, 0, disk_height/2])
p.changeDynamics(disk1Id, -1, lateralFriction=0, restitution=1)
p.changeDynamics(disk2Id, -1, lateralFriction=0, restitution=1)

# p.resetBaseVelocity(disk1Id, linearVelocity=[-3, 0, 0])

# Run the simulation

counter = 0
no_contact_counter = 0

bufferSize = 100
no_contact_tolerance = 50

preContact = False
bufferShrinked = False
preControl = 0
prev_pos_diff = (0, 0)
prev_robot_control = (0, 0)
candidate_count = 20
push_timestep = 20
estimation_threshold = 1000
dataset = []

timestep_counter = 0  # Initialize a counter to track the number of timesteps since the last position change

while (len(dataset) < bufferSize):

    # Make the object quasi-static
    p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0], angularVelocity = [0, 0, 0])
    object_pos, object_ori = p.getBasePositionAndOrientation(disk2Id)
    robot_actual_pos, _ = p.getBasePositionAndOrientation(disk1Id)
    # print(robot_actual_pos)
    # print("object_pos", object_pos)
    # pos_angle = random.uniform(-math.pi, math.pi)
    pos_angle = random.uniform(0, 2 * math.pi)

    robot_pos_x = object_pos[0] + 0.2 * math.cos(pos_angle)
    robot_pos_y = object_pos[1] + 0.2 * math.sin(pos_angle)
    # print("robot_pos", robot_pos_x, robot_pos_y)

    p.resetBasePositionAndOrientation(disk1Id, (robot_pos_x, robot_pos_y, disk_height/2), (1, 0, 0, 0))
    # p.resetBaseVelocity(disk1Id, linearVelocity=[-math.cos(angle), -math.sin(angle), 0])
    control_angle = random.uniform(-math.pi, math.pi)
    # control_angle = random.uniform(0, 2 * math.pi)
    # random_x_vel = random.uniform(-1, 1)
    # random_y_vel = random.uniform(-1, 1)

    control_x = math.cos(control_angle)
    control_y = math.sin(control_angle)
    # p.resetBaseVelocity(disk1Id, linearVelocity=[-math.cos(angle), -math.sin(angle), 0])
    for i in range (push_timestep):
        p.resetBaseVelocity(disk1Id, linearVelocity=[control_x, control_y, 0])
        p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(timeStep)

    end_pos, _ = p.getBasePositionAndOrientation(disk2Id)
    if (math.sqrt((end_pos[0] - object_pos[0]) ** 2 + (end_pos[1] - object_pos[1]) ** 2) > 0.0001):
        object_vel_angle = math.atan2(end_pos[1] - object_pos[1], end_pos[0] - object_pos[0])
        # if (object_vel_angle < 0):
        #     object_vel_angle -= 2 * math.pi
        #     control_angle -= 2 * math.pi
        dataset.append((pos_angle, control_angle, object_vel_angle))
        print(len(dataset))
    # object_vel_angle = math.atan2(end_pos[1] - object_pos[1], end_pos[0] - object_pos[0])
    # dataset.append((object_vel_angle, pos_angle, control_angle))
    # print(len(dataset))
data = np.array(dataset)
# data = np.load("concatenated_data.npy")
for element in data:
    if (element[2] < -math.pi / 2 and element[0] > math.pi / 2):
        element[2] += math.pi * 2
    if (element[2] > math.pi / 2 and element[0] < -math.pi / 2):
        element[2] -= math.pi * 2
    if (element[1] < math.pi / 2 and element[0] > math.pi / 2):
        element[1] += math.pi * 2
    if (element[1] > math.pi * 1.5 and element[2] < -math.pi):
        element[1] -= math.pi * 2
# np.save("cylinder_push.npy", data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Split the data into separate variables for clarity
object_vel_angles = data[:, 2]
pos_angles = data[:, 0]
control_angles = data[:, 1]

# Scatter plot
ax.scatter(object_vel_angles, pos_angles, control_angles)

# Label axes
ax.set_xlabel('Object Velocity Angle')
ax.set_ylabel('Position Angle')
ax.set_zlabel('Control Angle')

plt.show()

# k = 50

# kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

# Initial data collection over, start modelling and 
# eps = 0.2
# min_samples = 10
# dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
# clusters = kmeans.labels_

# filtered_data = data[clusters != -1]
# filtered_clusters = clusters[clusters != -1]

centroids = []

# Visualize the clusters
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'o']
# # unique_clusters = set(filtered_clusters)

# for i in range(k):
#     ax.scatter(data[clusters == i, 0], data[clusters == i, 1], data[clusters == i, 2], c=colors[i], label=f'Cluster {i+1}')
# for cluster in unique_clusters:
#     ax.scatter(filtered_data[filtered_clusters == cluster, 0], filtered_data[filtered_clusters == cluster, 1], filtered_data[filtered_clusters == cluster, 2], label=f'Cluster {cluster}')
# # for label in unique_labels:
# #     ix = np.where(clusters == label)
# #     ax.scatter(data[ix, 0], data[ix, 1], data[ix, 2], c='k' if label == -1 else colors[label % len(colors)], label='Noise' if label == -1 else f'Cluster {label + 1}')

# ax.set_xlabel('object_vel_angle')
# ax.set_ylabel('pos_angle')
# ax.set_zlabel('control_angle')
# ax.legend()

# plt.title('3D visualization of DBSCAN clustering')
# plt.show()

# # Linearize for each of the clusters
# models = []
# coefficients = []
# intercepts = []

# for cluster in unique_clusters:
#     cluster_data = filtered_data[filtered_clusters == cluster]
#     centroid = np.mean(cluster_data, axis=0)
#     centroids.append(centroid)
#     X = cluster_data[:, :-1]
#     y = cluster_data[:, -1]

#     model = LinearRegression().fit(X, y)
#     # model = Ridge().fit(X, y)
#     # model = ElasticNet.fit(X, y)
#     score = model.score(X, y)
#     print("Score", score)
#     models.append(model)

#     coefficients.append((model.coef_[0], model.coef_[1], model.intercept_))
k = 1

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
clusters = kmeans.labels_

# Visualization of the clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for cluster in range(k):
    # Select data points that belong to the current cluster
    cluster_data = data[clusters == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[cluster % len(colors)], label=f'Cluster {cluster}')
ax.set_xlabel('object_vel_angle')
ax.set_ylabel('pos_angle')
ax.set_zlabel('control_angle')
ax.legend()
plt.title('3D visualization of K-means clustering')
plt.show()

# Proceed with your linearization for each cluster as before
centroids = kmeans.cluster_centers_
models = []
coefficients = []
for cluster in range(k):
    cluster_data = data[clusters == cluster]
    X = cluster_data[:, :-1]
    y = cluster_data[:, -1]

    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    print("Cluster", cluster, "Score", score)
    models.append(model)

    coefficients.append((model.coef_[0], model.coef_[1], model.intercept_))
# print(len(models))
print(coefficients)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y', 'c']

for cluster in range(k):
    cluster_data = data[clusters == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[cluster % len(colors)], label=f'Cluster {cluster}')

# Plotting the planes
for cluster, (a, b, c) in enumerate(coefficients):
    # Generate grid over data range
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    
    # Calculate corresponding Z values
    Z = a * X + b * Y + c

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[cluster % len(colors)])

ax.set_xlabel('object_vel_angle')
ax.set_ylabel('pos_angle')
ax.set_zlabel('control_angle')
ax.legend()
plt.title('3D Visualization of Clustering with Linearized Planes')
plt.show()

# for cluster in unique_clusters:
#     cluster_data = filtered_data[filtered_clusters == cluster]
#     ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[cluster % len(colors)], label=f'Cluster {cluster}')



    # Plane equation: z = a*x + b*y + c
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
#     coef = coefficients[cluster]
#     Z = coef[0] * X + coef[1] * Y + coef[2]
#     ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[cluster % len(colors)])
# ax.set_zlim(-4, 4)
# ax.set_xlabel('object_vel_angle')
# ax.set_ylabel('pos_angle')
# ax.set_zlabel('control_angle')
# ax.legend()
# plt.title('3D Visualization of K-means Clustering with Linearized Planes (k=4)')
# plt.show()

p.resetBasePositionAndOrientation(disk1Id, (0.5, 0, disk_height/2), (1, 0, 0, 0))
p.resetBasePositionAndOrientation(disk2Id, (0.3, 0.1, disk_height/2), (1, 0, 0, 0))

# plt.ion()

counter = 0
while True:
    counter += 1
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y', 'c']
    
    p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
    object_pos, _ = p.getBasePositionAndOrientation(disk2Id)
    robot_pos, _ = p.getBasePositionAndOrientation(disk1Id)
    pos_angle = math.atan2((robot_pos[1] - object_pos[1]), (robot_pos[0] - object_pos[0]))
    if (pos_angle < 0):
        pos_angle += math.pi * 2

    # desired_object_vel_angle = math.atan2(goal_y_pos - object_pos[1], goal_x_pos - object_pos[0])
    desired_object_vel_angle = math.atan2(goal_y_pos - object_pos[1], goal_x_pos - object_pos[0])
    best_control_difference = 9999
    for i in range (100):
        control_angle = random.uniform(-math.pi, math.pi)
        new_data_point = [pos_angle, control_angle]
        new_object_velocity = models[0].predict([new_data_point])[0]
        if (abs(new_object_velocity - desired_object_vel_angle)) < best_control_difference:
            best_object_angle = abs(new_object_velocity - desired_object_vel_angle)
            best_control_angle = control_angle

    control_x = math.cos(best_control_angle)
    control_y = math.sin(best_control_angle)
    for i in range (push_timestep):
        p.resetBaseVelocity(disk1Id, linearVelocity=[control_x, control_y, 0])
        p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(timeStep)
