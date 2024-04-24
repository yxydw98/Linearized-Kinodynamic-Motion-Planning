import numpy as np
import sim


class Bounds(object):
    """
    Bounds of the state space for a motion planning problem.
    """
    
    def __init__(self, dim):
        """
        args: dim: The dimensionality of the state space.
                   Type: int
        """
        self.dim = dim
        # default bounds: [-1, 1] for all dimensions
        self.low = -np.ones(shape=(self.dim,))
        self.high = np.ones(shape=(self.dim,))

    def set_bounds(self, i, low, high):
        """
        Set bounds for one dimension of the state space.
        args:    i: The index of the dimension to set.
                    Type: int
               low: The lower bound.
              high: The upper bound.   
        """
        assert i >= 0 and i < self.dim
        self.low[i] = low
        self.high[i] = high

    def is_satisfied(self, state):
        stateVec = state["stateVec"]
        if not np.all(stateVec >= self.low):
            return False
        if not np.all(stateVec <= self.high):
            return False
        return True


class ProblemDefinition(object):
    """
    The definition of the motion planning problem to be solved.
    This includes necessary information and specifications of the problem 
    (e.g., start state, goal, bounds, etc).
    """

    def __init__(self, panda_sim):
        """
        args: panda_sim: The simulation environment of the Franka Panda robot based on pybullet.
                         Type: sim.PandaSim
        """
        self.panda_sim = panda_sim
        self.panda_sim.set_pdef(self)
        self.start_state = self.panda_sim.save_state()
        self.goal = None
        self.dim_state = sim.pandaNumDofs + 3 * self.panda_sim.num_objects # dimensionality of the state space
        self.dim_ctrl = 4 # dimensionality of the control space
        self.bounds_state = Bounds(self.dim_state) # bounds of the state space
        self.bounds_ctrl = Bounds(self.dim_ctrl) # bounds of the control space

    def get_state_dimension(self):
        return self.dim_state

    def get_control_dimension(self):
        return self.dim_ctrl

    def distance_func(self, stateVec1, stateVec2):
        """
        Calculate the Euclidean distance of two state vector(s).
        """
        dim = stateVec1.shape[-1]
        dists = np.linalg.norm(stateVec1.reshape(-1, dim) - stateVec2.reshape(-1, dim), axis=1)
        return dists

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    def get_start_state(self):
        return self.start_state
    
    def set_start_state(self, state_st):
        self.start_state = state_st

    def set_state_bounds(self, bounds):
        self.bounds_state = bounds

    def set_control_bounds(self, bounds):
        self.bounds_ctrl = bounds

    def is_state_valid(self, state):
        """
        Check if a state is valid or not.
        args: state: The query state of the system.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray of shape (self.dim_state,)}
        returns: Ture or False.
        """
        ########## TODO ##########
        if self.bounds_state.is_satisfied(state) == False:
            # print("is not satisfied")
            return False
        
        if self.panda_sim.is_collision(state):
            # print("is collison")
            return False
        
        ee_coordinates, _ = self.panda_sim.jac_solver.forward_kinematics(state["stateVec"])
        ee_coordinates = ee_coordinates + np.array([-0.4, -0.2, 0])
        # print(np.shape(ee_coordinates))
        if (ee_coordinates[0] <= -0.35 or ee_coordinates[1] <= -0.35 or ee_coordinates[0] >= 0.35 or ee_coordinates[1] >= 0.35):
            # print("is not in bounds")
            return False
        
        return True


        ##########################
    
    def is_state_high_quality(self, J):
        """
        Check if a state is high-quality enough or not for the task.
        args: J: The Jacobian matrix of the robot at the query state.
                 Type: numpy.ndarray of shape (6, 7)
        returns: Ture or False
        """
        ########## TODO ##########
        manipulability = np.sqrt(np.linalg.det(np.dot(J, J.T)))
        if manipulability < 0.005:
            # print("is not high quality", manipulability)
            return False
        # print("manipulability", manipulability)
        return True
    

        ##########################

    def propagate(self, nstate, control):
        self.panda_sim.restore_state(nstate)
        valid = self.panda_sim.execute(control)
        state = self.panda_sim.save_state()
        return state, valid
