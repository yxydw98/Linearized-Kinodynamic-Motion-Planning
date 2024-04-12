import time
import argparse
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import matplotlib.pyplot as plt
import numpy as np
import sim
from pdef import Bounds, ProblemDefinition
from goal import RelocateGoal, GraspGoal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import rrt
import utils
import opt
import math
import random

def normalize_vector(origin, target):
  vector_diff = np.array([target[0] - origin[0], target[1] - origin[1]])
  magnitude = np.linalg.norm(vector_diff)

  if magnitude > 0:
    normalized_vector = vector_diff / magnitude
  else:
    normalized_vector = vector_diff

  return normalized_vector

def get_pos_angle(eef_pos, object_pos):
  return utils.normalize_angle(math.atan2(eef_pos[1] - object_pos[1], eef_pos[0] - object_pos[0]))

def setup_pdef(panda_sim):
  pdef = ProblemDefinition(panda_sim)
  dim_state = pdef.get_state_dimension()
  dim_ctrl = pdef.get_control_dimension()

  # define bounds for state and control space
  bounds_state = Bounds(dim_state)
  for j in range(sim.pandaNumDofs):
    bounds_state.set_bounds(j, sim.pandaJointRange[j, 0], sim.pandaJointRange[j, 1])
  for j in range(sim.pandaNumDofs, dim_state):
    if ((j - sim.pandaNumDofs) % 3 == 2):
      bounds_state.set_bounds(j, -np.pi, np.pi)
    else:
      bounds_state.set_bounds(j, -0.3, 0.3)
  pdef.set_state_bounds(bounds_state)

  bounds_ctrl = Bounds(dim_ctrl)
  bounds_ctrl.set_bounds(0, -0.2, 0.2)
  bounds_ctrl.set_bounds(1, -0.2, 0.2)
  bounds_ctrl.set_bounds(2, -1.0, 1.0)
  bounds_ctrl.set_bounds(3, 0.4, 0.6)
  pdef.set_control_bounds(bounds_ctrl)
  return pdef


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=int, choices=[0, 1, 2, 3, 4, 5])
  args = parser.parse_args()

  # set up the simulation
  pgui = utils.setup_bullet_client(p.GUI)
  panda_sim = sim.PandaSim(pgui)

  if args.task == 0:
    utils.setup_cylinder_push(panda_sim) #cylinder radius = 0.02
    pdef = setup_pdef(panda_sim)
    object_pos = panda_sim.get_object_pos()
    eef_pos, _ = panda_sim.get_ee_pose()

    # Get contact with the object
    while (not panda_sim.in_collision_with_object()):
      control = normalize_vector(eef_pos, object_pos) / 100
      panda_sim.execute([control[0], control[1], 0, 1])

    # Generate a few random controls after getting contact to estimate the model
    dataset = []
    for _ in range (20):
      object_pos = panda_sim.get_object_pos()
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = get_pos_angle(eef_pos, object_pos)
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
      # print("normal", oppo_pos_angle, "pos_angle", pos_angle)
      control_angle = random.uniform(-1.2, 1.2)
      # control_angle = -1
      # control_angle = random.uniform(-math.pi, math.pi)

      control_x = math.cos(control_angle + oppo_pos_angle) / 100
      control_y = math.sin(control_angle + oppo_pos_angle) / 100
      # control_x = math.cos(control_angle) / 200
      # control_y = math.sin(control_angle) / 200
      # print("execution", control_angle + oppo_pos_angle, "control angle", control_angle)
      panda_sim.execute([control_x, control_y, 0, 1])
      end_object_pos = panda_sim.get_object_pos()
      object_vel_angle = math.atan2((end_object_pos[1] - object_pos[1]), (end_object_pos[0] - object_pos[0]))
      # print("result", object_vel_angle)
      object_vel_angle = utils.normalize_angle(object_vel_angle - oppo_pos_angle)
      dataset.append((control_angle, object_vel_angle))
      print(len(dataset))

    data = np.array(dataset)

    # Plot the data collected
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-math.pi, math.pi])
    ax.set_ylim([-math.pi, math.pi])

    control_angles = data[:, 0]
    object_vel_angles = data[:, 1]

    ax.scatter(control_angles, object_vel_angles)

    ax.set_xlabel('Control Angle')
    ax.set_ylabel('Object Velocity Angle')

    plt.show()

    # Linearize / Polynomial Regress the collected data
    x = data[:, 0]
    y = data[:, 1]

    linear_model = LinearRegression().fit(x.reshape(-1, 1), y)

    # Score of the linear model
    linear_score = r2_score(y, linear_model.predict(x.reshape(-1, 1)))
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.title('Linear Model Regression')
    plt.xlabel('X')
    plt.ylabel('Y')

    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = linear_model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, color='red', label='Linear Fit')
    plt.legend()
    plt.show()

    while True:
      object_pos = panda_sim.get_object_pos()
      #Check if the goal state is achieved
      if (utils.distance(object_pos, panda_sim.cylinder_push_goal) < 0.002):
        print("Goal reached!!!")
        panda_sim.execute([0, 0, 0, 100000])
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = math.atan2((eef_pos[1] - object_pos[1]), (eef_pos[0] - object_pos[0]))
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

      desired_object_vel_angle = math.atan2(panda_sim.cylinder_push_goal[1] - object_pos[1], panda_sim.cylinder_push_goal[0] - object_pos[0])
      desired_object_vel_angle = utils.normalize_angle(desired_object_vel_angle - oppo_pos_angle)

      x_range = np.linspace(-1, 1, 400)
      y_pred = linear_model.predict(x_range.reshape(-1, 1))

      differences = np.abs(y_pred - desired_object_vel_angle)

      min_diff_x = np.argmin(differences)
      x_closest = x_range[min_diff_x]
      print("x_closest = ", x_closest)
      x_closest += oppo_pos_angle

      control_x = math.cos(x_closest) / 200
      control_y = math.sin(x_closest) / 200
      panda_sim.execute([control_x, control_y, 0, 0.2])

  if args.task == 1:
    utils.setup_cube_push(panda_sim)
    pdef = setup_pdef(panda_sim)
    object_pos = panda_sim.get_object_pos()
    eef_pos, _ = panda_sim.get_ee_pose()

    # Get contact with the object
    while (not panda_sim.in_collision_with_object()):
      control = normalize_vector(eef_pos, object_pos) / 100
      panda_sim.execute([control[0], control[1], 0, 1])

    dataset = []
    for _ in range (20):
      object_pos = panda_sim.get_object_pos()
      object_ori = panda_sim.get_object_ori()
      print(object_ori)
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = get_pos_angle(eef_pos, object_pos)
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

      control_angle = random.uniform(-1.2, 1.2)
      execution_angle = control_angle + oppo_pos_angle + object_ori
      control_x = math.cos(execution_angle) / 200
      control_y = math.sin(execution_angle) / 200

      panda_sim.execute([control_x, control_y, 0, 1])

      end_object_pos = panda_sim.get_object_pos()
      object_vel_angle = math.atan2((end_object_pos[1] - object_pos[1]), (end_object_pos[0] - object_pos[0]))
      object_vel_angle = utils.normalize_angle(object_vel_angle - oppo_pos_angle - object_ori)
      dataset.append((control_angle, object_vel_angle))

      print(len(dataset))
    
    # data = np.array(dataset)
    data = np.load("20_data.npy")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-math.pi, math.pi])
    ax.set_ylim([-math.pi, math.pi])

    object_vel_angles = data[:, 1]
    control_angles = data[:, 0]

    ax.scatter(control_angles, object_vel_angles)
    ax.set_xlabel('Control Angle')
    ax.set_ylabel('Object Velocity Angle')
    
    plt.show()

    x = data[:, 0]
    y = data[:, 1]

    linear_model = LinearRegression().fit(x.reshape(-1, 1), y)

    # Score of the linear model
    linear_score = r2_score(y, linear_model.predict(x.reshape(-1, 1)))
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.title('Linear Model Regression')
    plt.xlabel('X')
    plt.ylabel('Y')

    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = linear_model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, color='red', label='Linear Fit')
    plt.legend()
    plt.show()

    while True:
      object_pos = panda_sim.get_object_pos()
      #Check if the goal state is achieved
      if (utils.distance(object_pos, panda_sim.cylinder_push_goal) < 0.002):
        print("Goal reached!!!")
        panda_sim.execute([0, 0, 0, 100000])
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = math.atan2((eef_pos[1] - object_pos[1]), (eef_pos[0] - object_pos[0]))
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

      desired_object_vel_angle = math.atan2(panda_sim.cylinder_push_goal[1] - object_pos[1], panda_sim.cylinder_push_goal[0] - object_pos[0])
      desired_object_vel_angle = utils.normalize_angle(desired_object_vel_angle - oppo_pos_angle)

      x_range = np.linspace(-1, 1, 400)
      y_pred = linear_model.predict(x_range.reshape(-1, 1))

      differences = np.abs(y_pred - desired_object_vel_angle)

      min_diff_x = np.argmin(differences)
      x_closest = x_range[min_diff_x]
      print("x_closest = ", x_closest)
      x_closest += oppo_pos_angle

      control_x = math.cos(x_closest) / 200
      control_y = math.sin(x_closest) / 200
      panda_sim.execute([control_x, control_y, 0, 0.2])





  if args.task == 5:
    pdef = setup_pdef(panda_sim)

    ctrls = [[0.02, 0, 0, 10],
             [0, 0.02, 0, 10],
             [-0.02, 0, 0, 10],
             [0, -0.02, 0, 10]]
    errs = []
    for _ in range(10):
      for ctrl in ctrls:
        wpts_ref = utils.extract_reference_waypoints(panda_sim, ctrl)
        wpts, _ = panda_sim.execute(ctrl)
        err_pos = np.mean(np.linalg.norm(wpts[:, 0:2] - wpts_ref[:, 0:2], axis=1))
        err_orn = np.mean(np.abs(wpts[:, 2] - wpts_ref[:, 2]))
        print("The average Cartesian error for executing the last control:")
        print("Position: %f meters\t Orientation: %f rads" % (err_pos, err_orn))
        errs.append([err_pos, err_orn])
    errs = np.array(errs)
    print("\nThe average Cartesian error for the entire exeution:")
    print("Position: %f meters\t Orientation: %f rads" % (errs[:, 0].mean(), errs[:, 1].mean()))


  else:
    # configure the simulation and the problem
    utils.setup_env(panda_sim)
    pdef = setup_pdef(panda_sim)

    # Task 2: Kinodynamic RRT Planning for Relocating
    if args.task == 2:
      goal = RelocateGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        while True:
          pass

    # Task 3: Kinodynamic RRT Planning for Grasping
    elif args.task == 3:
      goal = GraspGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        panda_sim.grasp()
        while True:
          pass

    # Task 4: Trajectory Optimization
    elif args.task == 4:
      ########## TODO ##########
      goal = RelocateGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found and Optimizing!")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan) 
        optimizer = opt.Optimization(pdef)
        plan = optimizer.path_optimization(plan)
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_optimized_plan(panda_sim, plan) 
        while True:
          pass
      pass

    
      ##########################
