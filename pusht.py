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
TId = p.loadURDF("objects/T.urdf", basePosition = [0, 0, 0])
T2Id = p.loadURDF("objects/T.urdf", basePosition = [0, 1.5, 0])
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

p.changeDynamics(TId, -1, lateralFriction=0, restitution=1)
p.changeDynamics(T2Id, -1, lateralFriction=0, restitution=1)

# p.resetBaseVelocity(disk1Id, linearVelocity=[0, 1, 0])

y_vel = 0
y2_vel = 0
while True:
    p.resetBaseVelocity(disk1Id, linearVelocity=[-0.2, y_vel, 0])
    p.resetBaseVelocity(disk2Id, linearVelocity=[-0.2, y2_vel, 0])
    contacts = p.getContactPoints(TId, disk1Id)

    obj_pos, obj_ori = p.getBasePositionAndOrientation(TId)
    rob_pos, rob_ori = p.getBasePositionAndOrientation(disk1Id)
    
    if contacts:
        # p.resetBaseVelocity(disk1Id, linearVelocity=[-0.2, -0.2, 0])
        y_vel = 0.2
        y2_vel = -0.2
    # print(obj_pos, obj_ori)
    p.stepSimulation()
    time.sleep(timeStep)
