import numpy as np
import jac
import pybullet as p
import math


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        targetcube_state = state["stateVec"][7:10]
        targetcube_pos = state["stateVec"][7:9]
        # print("targetcube_state", targetcube_state)
        ee_coordinates, ee_orientation = self.jac_solver.forward_kinematics(state["stateVec"])
        ee_coordinates[0] = ee_coordinates[0] - 0.4
        ee_coordinates[1] = ee_coordinates[1] - 0.2 
        ee_planar_position = ee_coordinates[:2]
        dist = np.linalg.norm(targetcube_pos - ee_planar_position)
        # print("distance", dist)
        # if (dist > 0.015):
        #     return False
        ee_axis, ee_angle = p.getAxisAngleFromQuaternion(ee_orientation)
        # print("ee_axis", ee_axis, "ee_angle", ee_angle)

        angle_difference = (ee_angle - targetcube_state[2]) % 1.57079632
        angle_difference = min(angle_difference, 1.57079632 - angle_difference)
        # print("angle difference", angle_difference)
        d1 = math.cos(angle_difference) * dist
        d2 = math.sin(angle_difference) * dist
        # print("d1", d1, "d2", d2)
        if (d1 > 0.01 or d2 > 0.02):
            return False
        if (angle_difference > 0.2):
            return False
        
        return True

        ##########################
        