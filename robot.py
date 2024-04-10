import pybullet as p
import pybullet_data
import numpy as np
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load environment and robot
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=1)

# Let the world run for a bit
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Specify desired end effector linear velocity (vx, vy, vz) in m/s
desired_velocity = np.array([0.01, 0.0, 0.0])  # Move along the X-axis

# Get the robot's joint states
num_joints = p.getNumJoints(robotId)
joint_states = p.getJointStates(robotId, range(num_joints))
joint_positions = np.array([x[0] for x in joint_states], dtype=np.float64)
joint_velocities = np.array([x[1] for x in joint_states], dtype=np.float64)
joint_accelerations = np.zeros(num_joints)  # Assuming static case for simplicity

# Calculate the Jacobian for the current joint configuration
# Ensure you provide the local position in the end effector link frame where you want to calculate the Jacobian.
end_effector_link_index = 7  # End effector link index for Franka Panda
local_position = [0, 0, 0]  # Position on the end effector where the Jacobian is computed, relative to the link frame

jacobian_linear, jacobian_angular = p.calculateJacobian(
    robotId,
    end_effector_link_index,
    local_position,
    joint_positions.tolist(),
    joint_velocities.tolist(),
    joint_accelerations.tolist()
)

# The Jacobian matrix might be redundant, calculate its pseudo-inverse for solving
jacobian_linear_np = np.array(jacobian_linear)
pseudo_inverse_jacobian = np.linalg.pinv(jacobian_linear_np)

# Solve for joint velocities
joint_velocities_target = np.dot(pseudo_inverse_jacobian, desired_velocity)

# Apply joint velocity commands
for i, velocity in enumerate(joint_velocities_target):
    p.setJointMotorControl2(bodyUniqueId=robotId,
                            jointIndex=i,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=velocity)

# Run the simulation for a short time to observe the movement
for _ in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()