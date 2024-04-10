import numpy as np
import copy
import sim
from goal import RelocateGoal
import rrt
import utils
import time
import pdef

class Optimization(object):
    def __init__(self, pdef):
        self.pdef = pdef
        self.cost_threshold = 50
        self.min_step = 20
        
    def path_optimization(self, plan, time_budget = 200.0, K = 10):
        start_time = time.time()    
        original_plan_dist = 0
        for i in range (1, len(plan)):
            original_plan_dist += np.linalg.norm(plan[i].state["stateVec"][:7] - plan[i - 1].state["stateVec"][:7])
        print("Original Distance = ", original_plan_dist)
        optimized_plan_dist = original_plan_dist
        while (time.time() - start_time < time_budget):
            #Generate k candidate plans and choose the best from it
            best_candidate = None
            for k in range (K):
                candidate_plan = self.add_noise(plan)
                candidate_plan_dist = 0
                #Calculate the distance of this candidate plan
                for i in range (1, len(candidate_plan)):
                    state, valid = self.pdef.propagate(candidate_plan[i - 1].state, candidate_plan[i].control)
                    if valid and self.pdef.is_state_valid(state):
                        candidate_plan[i].state = state
                    else:
                        break
                    candidate_plan_dist += np.linalg.norm(state["stateVec"][:7] - candidate_plan[i - 1].state["stateVec"][:7])
                if (self.pdef.goal.is_satisfied(state) and candidate_plan_dist < optimized_plan_dist):
                    best_candidate = candidate_plan
                    optimized_plan_dist = candidate_plan_dist
                    print("Optimized Distance", optimized_plan_dist)
            if best_candidate is not None:
                plan = best_candidate
                print("Best plan Updated!", optimized_plan_dist)
        return plan
# def is_satisfied(state):
#     stateVec = state["stateVec"]
#     x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
#     if np.linalg.norm([x_tgt - 0.2, y_tgt + 0.2]) < 0.1:
#         return True
#     else:
#         return False       
                
                

    def add_noise(self, plan, mu=0, sigma=0.01):
        noisy_plan = copy.deepcopy(plan)
        control_sequence = []
        for node in noisy_plan[1:]:
            new_control = node.control + np.random.normal(mu, sigma, node.control.shape)
            node.set_control(new_control)
            control_sequence.append(new_control)
        # print(control_sequence)
        return noisy_plan


