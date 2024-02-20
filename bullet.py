import pybullet as p
import time
import pybullet_data
import random
import numpy as np

# def 
# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

timeStep = 1.0/240
p.setTimeStep(timeStep)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=90, cameraPitch=-80, cameraTargetPosition=[0.5, 0, 0.75])


# Load a plane and a table
planeId = p.loadURDF("plane.urdf")
tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])

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
p.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[goal_x_pos, goal_y_pos, table_height])

# Position the disks on the table
# Adjust z-coordinate to be table_height plus half of disk height
disk1Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk1, basePosition=[0.5, 0, table_height + disk_height/2])
disk2Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk2, basePosition=[0.3, 0, table_height + disk_height/2])
p.changeDynamics(disk1Id, -1, lateralFriction=1, restitution=1)
p.changeDynamics(disk2Id, -1, lateralFriction=2, restitution=1)

# p.resetBaseVelocity(disk1Id, linearVelocity=[-3, 0, 0])

# Run the simulation

counter = 0
no_contact_counter = 0

bufferSize = 200
no_contact_tolerance = 50

preContact = False
bufferShrinked = False
preControl = 0
prev_pos_diff = 0
prev_vel_diff = 0

dataset = []

while True:
    # Get the state of the robot
    disk1_pos, disk1_ori = p.getBasePositionAndOrientation(disk1Id)
    disk1_vel, disk1_ang_vel = p.getBaseVelocity(disk1Id)

    # Get the state of the object
    disk2_pos, disk2_ori = p.getBasePositionAndOrientation(disk2Id)
    disk2_vel, disk2_ang_vel = p.getBaseVelocity(disk2Id)
  
    #Get the relative position and velocity of the two disks
    disk_pos_diff = (disk2_pos[0] - disk1_pos[0], disk2_pos[1] - disk1_pos[1])
    disk_vel_diff = (disk2_vel[0] - disk1_vel[0], disk2_vel[1] - disk1_vel[1])
    
    # Print the states or use them for further calculations
    # print("Disk 1 Position:", disk1_pos, "Orientation:", disk1_ori, "Velocity:", disk1_vel, "Angular Velocity:", disk1_ang_vel)
    # print("Disk 2 Position:", disk2_pos, "Orientation:", disk2_ori, "Velocity:", disk2_vel, "Angular Velocity:", disk2_ang_vel)
    contacts = p.getContactPoints(disk1Id, disk2Id)

    rand_x_vel = random.random() / 2
    rand_y_vel = random.uniform(-1, 1)
    
    print("timestep", counter)

    # To make the environment quosi-static
    p.resetBaseVelocity(disk2Id, linearVelocity=[0, 0, 0])

    # Initialize the buffer by perturbation
    if (len(dataset) < bufferSize and counter > 0):
        p.resetBaseVelocity(disk1Id, linearVelocity=[-rand_x_vel, rand_y_vel, 0])
        if preContact: # Add into the dataset only if the velocity is caused by direct contact
            dataset.append((prev_vel_diff, disk2_vel[:2], prev_pos_diff))
        # print("current reset velocity = ", -rand_x_vel)

    # prev_disk1_vel = disk1_vel
    prev_pos_diff = disk_pos_diff
    # prev_vel_diff = disk_vel_diff
    prev_vel_diff = (rand_x_vel, rand_y_vel)

    # Print relative position and velocity
    print("Relative Position: ", disk_pos_diff)
    print("Robot Velocity", disk1_vel)
    print("Object Velocity", disk2_vel)
          

    # detect if there are contacts at the current stage and populate the info to the next timestep
    if contacts:
        print("Contact detected")
        preContact = True
        no_contact_counter = 0
    else:
        print("No contact")
        preContact = False
        no_contact_counter += 1
    counter += 1

    # Enforce contact if no contact has been detected for a period of time
    if (no_contact_counter >= no_contact_tolerance and preContact == False):
        print("Enforcing contacts")
        # Define the contact enforcement velocity as a normalized vector of the relative position
        
        contact_enforcement_vel_array = np.array(disk_pos_diff)
        magnitude = np.linalg.norm(contact_enforcement_vel_array)
        print("the current magnitude of the enforcement velocity", magnitude)
        normalized_contact_enforcement_vel = contact_enforcement_vel_array / magnitude
        print("the current contact_enforcement_vel", normalized_contact_enforcement_vel)
        normalized_contact_enforcement_vel = normalized_contact_enforcement_vel * 0.5
        p.resetBaseVelocity(disk1Id, linearVelocity=[normalized_contact_enforcement_vel[0], normalized_contact_enforcement_vel[1], 0])
        # if not bufferShrinked:
        #     bufferSize = len(dataset) // 2
        #     dataset = dataset[bufferSize:]
        #     bufferShrinked = True
        # Continue the simulation with only contact enforcement
        p.stepSimulation()
        time.sleep(timeStep)
        continue
        


    # start_time = time.time()
    # update the dataset with recent contact dynamics
    if (len(dataset) >= bufferSize):
        if preContact:
            dataset.pop(0)
            dataset.append((prev_vel_diff, disk2_vel[:2], prev_pos_diff))

        vel_diff = np.array([item[0] for item in dataset])
        object_vel = np.array([item[1] for item in dataset])
        pos_diff = np.array([item[2] for item in dataset])
        
        X = np.hstack((object_vel, pos_diff, np.ones((object_vel.shape[0], 1))))
        coefficients_x1 = np.linalg.lstsq(X, vel_diff[:, 0], rcond=None)[0]
        coefficients_x2 = np.linalg.lstsq(X, vel_diff[:, 1], rcond=None)[0]

        A_x1, A_x2, B_x1, B_x2, C1 = coefficients_x1
        A_y1, A_y2, B_y1, B_y2, C2 = coefficients_x2
        print(coefficients_x1)
        print(coefficients_x2)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time for linearization: ", elapsed_time) 

    # execute greedy control based on the current model
        desired_object_vel = np.array([goal_x_pos - disk2_pos[0], goal_y_pos - disk2_pos[1]])
        magnitude = np.linalg.norm(desired_object_vel)
        normalized_desired_object_vel = desired_object_vel / magnitude
        desired_object_vel = normalized_desired_object_vel * 0.05

        print("desired velocity", desired_object_vel)
        desired_object_vel = desired_object_vel.reshape(1, -1)
        disk_pos_diff_array = np.array(disk_pos_diff)
        disk_pos_diff_array = disk_pos_diff_array.reshape(1, -1)
        current_step = np.hstack((desired_object_vel, disk_pos_diff_array, np.ones((desired_object_vel.shape[0], 1))))

        predicted_vel_diff_x = np.dot(current_step, coefficients_x1)
        predicted_vel_diff_y = np.dot(current_step, coefficients_x2)

        print(predicted_vel_diff_x)

        disk1_execute_x = disk2_vel[0] - predicted_vel_diff_x[0]
        disk1_execute_y = disk2_vel[1] - predicted_vel_diff_y[0]

        execution_velocity = np.array([disk1_execute_x, disk1_execute_y])
        magnitude = np.linalg.norm(desired_object_vel)
        normalized_execution_velocity = execution_velocity / magnitude * 0.05
        disk1_execute_x = normalized_execution_velocity[0]
        disk1_execute_y = normalized_execution_velocity[1]
        # p.resetBaseVelocity(disk1Id, linearVelocity=[disk1_execute_x, disk1_execute_y, 0])

        p.resetBaseVelocity(disk1Id, linearVelocity=[disk1_execute_x, disk1_execute_y, 0])

    p.stepSimulation()
    time.sleep(timeStep)
