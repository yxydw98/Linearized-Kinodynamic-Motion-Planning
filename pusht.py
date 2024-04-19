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
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


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
obj_file = "objects/T.obj"
visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName=obj_file,
                                      rgbaColor=[1, 0, 0, 1],
                                      meshScale=[2, 2, 2])

collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=obj_file,
                                            meshScale=[2, 2, 2])

TId = p.createMultiBody(baseMass=1,
                           baseInertialFramePosition=[0,0,0],
                           baseCollisionShapeIndex=collision_shape_id,
                           baseVisualShapeIndex=visual_shape_id,
                           basePosition=[0,0,0],
                           baseOrientation=[0,0,0,1])

T2Id = p.createMultiBody(baseMass=1,
                           baseInertialFramePosition=[0,0,0],
                           baseCollisionShapeIndex=collision_shape_id,
                           baseVisualShapeIndex=visual_shape_id,
                           basePosition=[0,1.5,0],
                           baseOrientation=[0,0,0,1])
# p.changeDynamics(bodyUniqueId=TId,
#                  linkIndex=-1,  # -1 refers to the base link
#                  lateralFriction=0.5)

# p.changeDynamics(bodyUniqueId=T2Id,
#                  linkIndex=-1,  # -1 refers to the base link
#                  lateralFriction=0.5)
# tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])

# Assuming the table height is around 0.7 meters
table_height = 0.625

# Dimensions for the disks
disk_mass = 10
disk_radius = 0.08
disk_height = 0.02

goal_x_pos = -0.5
goal_y_pos = 0.25

# Create two disks (as cylinders)
disk1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)
disk2 = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)

visGoalID = p.createVisualShape(p.GEOM_CYLINDER,radius=0.1, length=1e-3,rgbaColor=[1.0, 0.0, 0.0, 1.0])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[goal_x_pos, goal_y_pos, 0])
disk1Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk1, basePosition=[0.5, 0, disk_height/2])
disk2Id = p.createMultiBody(baseMass=disk_mass, baseCollisionShapeIndex=disk1, basePosition=[0.5, 1.5, disk_height/2])

p.changeDynamics(TId, -1, lateralFriction=0.5, restitution=1)
p.changeDynamics(T2Id, -1, lateralFriction=0.5, restitution=1)

# p.resetBaseVelocity(disk1Id, linearVelocity=[0, 1, 0])

y_vel = 0
y2_vel = 0
counter = 0
while True:
    counter += 1
    p.resetBaseVelocity(disk1Id, linearVelocity=[-2, y_vel, 0])
    p.resetBaseVelocity(disk2Id, linearVelocity=[-2, y2_vel, 0])
    contacts = p.getContactPoints(TId, disk1Id)

    obj_pos, obj_ori = p.getBasePositionAndOrientation(TId)
    obj2_pos, obj2_ori = p.getBasePositionAndOrientation(T2Id)
    rob_pos, rob_ori = p.getBasePositionAndOrientation(disk1Id)
    
    if contacts:
        # p.resetBaseVelocity(disk1Id, linearVelocity=[-0.2, -0.2, 0])
        y_vel = 2
        y2_vel = -2
    # print(obj_pos, obj_ori)
        
    if (counter == 1000):
        print("Object1 position", obj_pos, "Object2 position", obj2_pos)
    p.stepSimulation()
    time.sleep(timeStep)
