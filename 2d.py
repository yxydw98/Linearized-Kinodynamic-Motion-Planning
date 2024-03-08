import pybullet as p
import time
import pybullet_data
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator, RegularGridInterpolator
from sklearn.metrics import mean_squared_error

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
disk_mass = 10
disk_radius = 0.1
disk_height = 0.02

goal_x_pos = -0.5
goal_y_pos = 0.25

# Create two disks (as cylinders)
disk1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)
disk2 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)

visGoalID = p.createVisualShape(p.GEOM_CYLINDER,radius=0.1, length=1e-3,rgbaColor=[1.0, 0.0, 0.0, 1.0])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[goal_x_pos, goal_y_pos, table_height])

# Position the disks on the table
# Adjust z-coordinate to be table_height plus half of disk height
disk1Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk1, basePosition=[0.5, 0, disk_height/2])
disk2Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk2, basePosition=[0.3, 0, disk_height/2])
p.changeDynamics(disk1Id, -1, lateralFriction=0, restitution=1)
p.changeDynamics(disk2Id, -1, lateralFriction=0, restitution=1)

# p.resetBaseVelocity(disk1Id, linearVelocity=[-3, 0, 0])

# Run the simulation

counter = 0
no_contact_counter = 0

bufferSize = 500
no_contact_tolerance = 50

preContact = False
bufferShrinked = False
preControl = 0
prev_pos_diff = (0, 0)
prev_robot_control = (0, 0)
candidate_count = 20

dataset = []

timestep_counter = 0  # Initialize a counter to track the number of timesteps since the last position change

while (len(dataset) < bufferSize):
    if timestep_counter == 0:
        # Make the object quasi-static
        p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
        object_pos, object_ori = p.getBasePositionAndOrientation(disk2Id)
        robot_actual_pos, _ = p.getBasePositionAndOrientation(disk1Id)
        # print(robot_actual_pos)
        # print("object_pos", object_pos)
        angle = random.uniform(0, 2 * math.pi)

        robot_pos_x = object_pos[0] + 0.2 * math.cos(angle)
        robot_pos_y = object_pos[1] + 0.2 * math.sin(angle)
        # print("robot_pos", robot_pos_x, robot_pos_y)

        p.resetBasePositionAndOrientation(disk1Id, (robot_pos_x, robot_pos_y, disk_height/2), (1, 0, 0, 0))
        # p.resetBaseVelocity(disk1Id, linearVelocity=[-math.cos(angle), -math.sin(angle), 0])
        random_x_vel = random.uniform(-1, 1)
        random_y_vel = random.uniform(-1, 1)
    # p.resetBaseVelocity(disk1Id, linearVelocity=[-math.cos(angle), -math.sin(angle), 0])
    p.resetBaseVelocity(disk1Id, linearVelocity=[random_x_vel, random_y_vel, 0])
    p.stepSimulation()
    time.sleep(timeStep)

    # if timestep_counter == 0:
    #     # Reset the velocity to simulate control without changing position
    #     p.resetBaseVelocity(disk1Id, linearVelocity=[0, 0, 0])
    if timestep_counter != 0:
        robot_vel = (random_x_vel, random_y_vel)
        robot_pos, _ = p.getBasePositionAndOrientation(disk1Id)
        object_vel, _ = p.getBaseVelocity(disk2Id)
        object_pos, _ = p.getBasePositionAndOrientation(disk2Id)
        relative_pos = (object_pos[0] - robot_pos[0], object_pos[1] - robot_pos[1])
        if (math.sqrt(object_vel[0] ** 2 + object_vel[1] ** 2) > 0.001):
            dataset.append((robot_vel, object_vel[:2], relative_pos))
        print("object_velocity", object_vel[:2])

    timestep_counter += 1

    if timestep_counter >= 3:
        timestep_counter = 0  # Reset the counter after 30 timesteps to update the position again
data = np.array(dataset)

control = np.array([item[0] for item in dataset])
object_vel = np.array([item[1] for item in dataset])
pos_diff = np.array([item[2] for item in dataset])
control_flat = control.flatten()
object_vel_flat = object_vel.flatten()
pos_diff_flat = pos_diff.flatten()

# interpolator = LinearNDInterpolator(list(zip(object_vel_flat, pos_diff_flat)), control_flat)
# interpolator = NearestNDInterpolator(list(zip(object_vel_flat, pos_diff_flat)), control_flat)
# interpolator = CloughTocher2DInterpolator(list(zip(object_vel_flat, pos_diff_flat)), control_flat)
interpolator = CloughTocher2DInterpolator(list(zip(object_vel_flat, pos_diff_flat)), control_flat)
print("Interpolation done")

# control_predicted = interpolator(object_vel, pos_diff)
# mse = mean_squared_error(control, control_predicted)
# print("Score of the model", mse)

x_vis, y_vis = np.meshgrid(np.linspace(0, 2, 50), np.linspace(0, 2, 50))
z_vis = interpolator(x_vis, y_vis)

# Plotting
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(object_vel_flat, pos_diff_flat, control_flat, color='r', label='Original Data Points')
ax.plot_surface(x_vis, y_vis, z_vis, cmap='viridis', edgecolor='none', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Piecewise Linear Interpolation Over a 2D Dataset')
ax.legend()
plt.show()

p.resetBasePositionAndOrientation(disk1Id, (0.5, 0, disk_height/2), (1, 0, 0, 0))
p.resetBasePositionAndOrientation(disk2Id, (0.3, 0, disk_height/2), (1, 0, 0, 0))
while True:
    p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])
    object_pos, _ = p.getBasePositionAndOrientation(disk2Id)
    robot_pos, _ = p.getBasePositionAndOrientation(disk1Id)
    relative_pos = (object_pos[0] - robot_pos[0], object_pos[1] - robot_pos[1])

    desired_object_vel = np.array([goal_x_pos - object_pos[0], goal_y_pos - object_pos[1]])

    magnitude = np.linalg.norm(desired_object_vel)
    normalized_desired_object_vel = desired_object_vel / magnitude
    desired_object_vel = normalized_desired_object_vel * 0.5

    relative_pos = np.array(relative_pos)
    print(desired_object_vel)
    print(relative_pos)
    control_predicted = interpolator(desired_object_vel, relative_pos)
    print("predicted_control", control_predicted)
    print(control_predicted.shape)
    p.resetBaseVelocity(disk1Id, linearVelocity=[control_predicted[0], control_predicted[1], 0])
    p.stepSimulation()
    time.sleep(timeStep)
                                    
# while True:
#     # Get the state of the robot
#     disk1_pos, disk1_ori = p.getBasePositionAndOrientation(disk1Id)
#     disk1_vel, disk1_ang_vel = p.getBaseVelocity(disk1Id)

#     # Get the state of the object
#     disk2_pos, disk2_ori = p.getBasePositionAndOrientation(disk2Id)
#     disk2_vel, disk2_ang_vel = p.getBaseVelocity(disk2Id)
  
#     #Get the relative position and velocity of the two disks
#     disk_pos_diff = (disk2_pos[0] - disk1_pos[0], disk2_pos[1] - disk1_pos[1])

#     # Print the states or use them for further calculations
#     # print("Disk 1 Position:", disk1_pos, "Orientation:", disk1_ori, "Velocity:", disk1_vel, "Angular Velocity:", disk1_ang_vel)
#     # print("Disk 2 Position:", disk2_pos, "Orientation:", disk2_ori, "Velocity:", disk2_vel, "Angular Velocity:", disk2_ang_vel)
#     contacts = p.getContactPoints(disk1Id, disk2Id)

#     rand_x_vel = -random.random() / 2
#     rand_y_vel = random.uniform(-1, 1)
    
#     print("timestep", counter)

#     # To make the environment quosi-static
#     p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])

#     # Initialize the buffer by perturbation
#     if (len(dataset) < bufferSize):
#         p.resetBaseVelocity(disk1Id, linearVelocity=[rand_x_vel, rand_y_vel, 0])
#         if preContact: # Add into the dataset only if the velocity is caused by direct contact
#             dataset.append((prev_robot_control, disk2_vel[:2], prev_pos_diff))
#             prev_pos_diff = disk_pos_diff
#             prev_robot_control = (rand_x_vel, rand_y_vel)
#         # print("current reset velocity = ", -rand_x_vel)

#     # prev_disk1_vel = disk1_vel
    
#     # prev_robot_control = disk_control
    

#     # Print relative position and velocity
#     print("Relative Position: ", disk_pos_diff)
#     print("Robot Velocity", disk1_vel)
#     print("Object Velocity", disk2_vel)
          

#     # detect if there are contacts at the current stage and populate the info to the next timestep
#     if contacts:
#         print("Contact detected")
#         preContact = True
#         no_contact_counter = 0
#     else:
#         print("No contact")
#         preContact = False
#         no_contact_counter += 1
#     counter += 1

#     # Enforce contact if no contact has been detected for a period of time
#     # if (no_contact_counter >= no_contact_tolerance and preContact == False):
#     #     ## Naive heuristic for enforcing contacts
#     #     print("Enforcing contacts")
#     #     # Define the contact enforcement velocity as a normalized vector of the relative position
        
#     #     contact_enforcement_vel_array = np.array(disk_pos_diff)
#     #     magnitude = np.linalg.norm(contact_enforcement_vel_array)
#     #     print("the current magnitude of the enforcement velocity", magnitude)
#     #     normalized_contact_enforcement_vel = contact_enforcement_vel_array / magnitude
#     #     print("the current contact_enforcement_vel", normalized_contact_enforcement_vel)
#     #     normalized_contact_enforcement_vel = normalized_contact_enforcement_vel * 0.5
#     #     p.resetBaseVelocity(disk1Id, linearVelocity=[normalized_contact_enforcement_vel[0], normalized_contact_enforcement_vel[1], 0])
#     #     # if not bufferShrinked:
#     #     #     bufferSize = len(dataset) // 2
#     #     #     dataset = dataset[bufferSize:]
#     #     #     bufferShrinked = True
#     #     # Continue the simulation with only contact enforcement
#     #     p.stepSimulation()
#     #     time.sleep(timeStep)
#     #     continue
        
#         ## Teleport to the best sampled position
#         # Sample k possible positions around the current robot, and use the current linearized function to predict which is the best velocity
#         # best_velocity = 10000
#         # for i in range (0, candidate_count):
#         #     x_alteration = random.uniform(-1, 1)
#         #     y_alteration = random.uniform(-1, 1)
            
#         #     candidate_robot_pos_x = x_alteration + disk1_pos[0]
#         #     candidate_robot_pos_y = y_alteration + disk1_pos[1]

#         #     temp_dist = math.sqrt((candidate_robot_pos_x - disk2_pos[0]) ** 2 + (candidate_robot_pos_y - disk2_pos[1]) ** 2)
#         #     if (temp_dist < 0.2): # The two disks are overlapping
#         #         i -= 1
#         #     else:
#         #         current_velocity = 
#         #         best_velocity = min(best_velocity, current_velocity)
                       




#     # update the dataset with recent contact dynamics

#     if (len(dataset) >= bufferSize):
#         start_time = time.time()
#         if preContact:
#             dataset.pop(0)
#             dataset.append((prev_robot_control, disk2_vel[:2], prev_pos_diff))
#             # print(dataset)



#         features = np.hstack([object_vel, pos_diff, np.ones((bufferSize, 1))])
#         targets = control

#         model = LinearRegression().fit(features, targets)
#         score = model.score(features, targets)
#         print("The model score is", score)

#         # coefficients = model.coef_
#         # intercept = model.intercept_

#         # A_B = coefficients[:, :-1]

#         # A = A_B[:, :2]
#         # B = A_B[:, 2:]
#         # C = intercept
#         # print("A", A, "B", B, "C", C)

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print("Elapsed time for linearization: ", elapsed_time) 

#     # execute greedy control based on the current model


#         print("desired velocity", desired_object_vel)
#         # desired_object_vel = desired_object_vel.reshape(1, -1)
#         # disk_pos_diff_array = np.array(disk_pos_diff)
#         # disk_pos_diff_array = disk_pos_diff_array.reshape(1, -1)
#         # current_step = np.hstack((desired_object_vel, disk_pos_diff_array, np.ones((desired_object_vel.shape[0], 1))))

#         # predicted_control_x = np.dot(current_step, coefficients_x1)
#         # predicted_control_y = np.dot(current_step, coefficients_x2)
#         disk_pos_diff_array = np.array(disk_pos_diff)
#         new_features = np.hstack([desired_object_vel, disk_pos_diff_array, np.ones(1)])
#         predicted_control = model.predict(new_features.reshape(1, -1))

#         print("predicted robot velocity", predicted_control)
        
#         disk1_execute_x = predicted_control[0][0]
#         disk1_execute_y = predicted_control[0][1]

#         # execution_velocity = np.array([disk1_execute_x, disk1_execute_y])
#         # magnitude = np.linalg.norm(desired_object_vel)
#         # normalized_execution_velocity = execution_velocity / magnitude * 0.05
#         # disk1_execute_x = normalized_execution_velocity[0]
#         # disk1_execute_y = normalized_execution_velocity[1]
#         # p.resetBaseVelocity(disk1Id, linearVelocity=[disk1_execute_x, disk1_execute_y, 0])

#         p.resetBaseVelocity(disk1Id, linearVelocity=[disk1_execute_x, disk1_execute_y, 0])
#         prev_pos_diff = disk_pos_diff
#         prev_robot_control = (disk1_execute_x, disk1_execute_y)

#     p.stepSimulation()
#     time.sleep(timeStep)