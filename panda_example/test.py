import pybullet as p

# Assuming 'robot' is the ID of your loaded robot
p.connect(p.GUI)
robot = p.loadURDF("assets/franka_panda/panda.urdf")

# Get the number of joints
num_joints = p.getNumJoints(robot)

# Loop through all joints to print their names and indices
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot, joint_index)
    joint_name = joint_info[1].decode('utf-8')  # Joint name
    link_name = joint_info[12].decode('utf-8')  # Link name (linkIndex = joint_index + 1 for most configurations)
    print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Link Name: {link_name}")

# To find and print the name of a specific link by index
specific_link_index = 9  # Example: linkIndex 11
specific_link_name = p.getJointInfo(robot, specific_link_index)[12].decode('utf-8')
print(f"The name of link with index {specific_link_index} is: {specific_link_name}")