import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)
    

class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None # the control asscoiated with this node
        self.parent = None # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode
    

class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler

    def solve(self, time_budget):
        """
        The main algorithm of Kinodynamic RRT.
        args:  time_budget: The planning time budget (in seconds).
        returns: is_solved: True or False.
                      plan: The motion plan found by the planner,
                            represented by a sequence of tree nodes.
                            Type: a list of rrt.Node
        """
        ########## TODO ##########
        solved = False
        plan = None

        start_time = time.time()
        start_node = Node(self.pdef.get_start_state())
        self.tree.add(start_node)
        
        while (time.time() - start_time < time_budget):
            rstateVec = self.state_sampler.sample()
            nearestNode = self.tree.nearest(rstateVec)
            # print(nearestNode)
            rcontrol, new_state = self.control_sampler.sample_to(nearestNode, rstateVec, 1)
            if (rcontrol is None or new_state is None):
                continue
        
            
            # new_stateVec = self.pdef.propagate(nearestNode.state, rcontrol)

            new_node = Node(new_state)
            new_node.set_control(rcontrol)
            new_node.set_parent(nearestNode)
            self.tree.add(new_node)

            # Check if the new node is close to the goal
            if self.pdef.goal.is_satisfied(new_node.state):
            # Construct the path from the start to the goal
                plan = []
                while new_node is not None:
                    plan.append(new_node)
                    new_node = new_node.get_parent()
                plan.reverse()
                solved = True
                break
        ##########################


        return solved, plan
