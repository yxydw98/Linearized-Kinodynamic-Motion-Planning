import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc

class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        J = np.zeros(shape=(6, 7))
        delta = 0.01
        for i in range(7):
            # Create a copy of the joint values and perturb the i-th joint
            joint_values_perturbed = joint_values.copy()
            joint_values_perturbed[i] += delta
        
            # Calculate the forward kinematics for the perturbed joint values
            pos_perturbed, quat_perturbed = self.forward_kinematics(joint_values_perturbed)
        
            # Get the original position and quaternion
            pos, quat = self.forward_kinematics(joint_values)
        
            # Calculate the difference in position and orientation
            pos_diff = (pos_perturbed - pos) / delta
            q_diff = self.bullet_client.getDifferenceQuaternion(quat, quat_perturbed)
            axis, angle = self.bullet_client.getAxisAngleFromQuaternion(q_diff)

            J[3:, i] = np.array(axis) * angle / delta
            J[:3, i] = pos_diff

        return J

