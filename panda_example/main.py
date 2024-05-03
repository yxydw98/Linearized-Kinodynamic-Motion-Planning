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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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
  parser.add_argument("--task", type=int, choices=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  parser.add_argument("--pretraining", action="store_true", help="Enable pretraining mode")
  parser.add_argument("--shift_angle", type=float, default=0.0, help="Shift angle from directly opposite the goal")
  parser.add_argument("--save_data", action="store_true", help="Save the data for this round of training")

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
        time.sleep(10000)
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
    
    data = np.array(dataset)
    # data = np.load("20_data.npy")
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
        time.sleep(10000)
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


  if args.task == 2:
    utils.setup_triangle_push(panda_sim)
    pdef = setup_pdef(panda_sim)
    object_pos = panda_sim.get_object_pos()
    eef_pos, _ = panda_sim.get_ee_pose()

    # Get contact with the object
    while (not panda_sim.in_collision_with_object()):
      control = normalize_vector(eef_pos, object_pos) / 100
      panda_sim.execute([control[0], control[1], 0 , 1])
    print("Contact detected")

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
    
    # plt.show()

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


    while True:
      object_pos = panda_sim.get_object_pos()
      #Check if the goal state is achieved
      if (utils.distance(object_pos, panda_sim.cylinder_push_goal) < 0.01):
        print("Goal reached!!!")
        time.sleep(10000)
      else:
        print("distance to goal", utils.distance(object_pos, panda_sim.cylinder_push_goal))
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

if args.task == 3:
    utils.setup_dynamic_cylinder_push(panda_sim, 0.01) #cylinder radius = 0.02
    pdef = setup_pdef(panda_sim)
    object_pos = panda_sim.get_object_pos()
    eef_pos, _ = panda_sim.get_ee_pose()
    dataset = []
    for _ in range (10):
    # Get contact with the object
      while (not panda_sim.in_collision_with_object()):
        object_pos = panda_sim.get_object_pos()
        eef_pos, _ = panda_sim.get_ee_pose()
        control = normalize_vector(eef_pos, object_pos) / 100
        panda_sim.execute([control[0], control[1], 0, 1.2])

    # Generate a few random controls after getting contact to estimate the model
      object_pos = panda_sim.get_object_pos()
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = get_pos_angle(eef_pos, object_pos)
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
      # print("normal", oppo_pos_angle, "pos_angle", pos_angle)
      control_angle = random.uniform(-1.2, 1.2)
      # control_angle = -1
      # control_angle = random.uniform(-math.pi, math.pi)

      control_x = math.cos(control_angle + oppo_pos_angle) / 8000
      control_y = math.sin(control_angle + oppo_pos_angle) / 8000
      # control_x = math.cos(control_angle) / 200
      # control_y = math.sin(control_angle) / 200
      # print("execution", control_angle + oppo_pos_angle, "control angle", control_angle)
      panda_sim.execute([control_x, control_y, 0, 0.5])
      current_object_velocity, _ = panda_sim.get_object_velocity()
      while (math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2 + current_object_velocity[2] ** 2) > 0.0001):
        current_object_velocity, _ = panda_sim.get_object_velocity()
        panda_sim.execute([0, 0, 0, 1])
      end_object_pos = panda_sim.get_object_pos()
      object_vel_angle = math.atan2((end_object_pos[1] - object_pos[1]), (end_object_pos[0] - object_pos[0]))
      # print("result", object_vel_angle)
      object_vel_angle = utils.normalize_angle(object_vel_angle - oppo_pos_angle)
      dataset.append((control_angle, object_vel_angle))
      print(len(dataset))

    # data = np.array(dataset)
    data = np.load("20_data.npy")
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
        time.sleep(10000)
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

      control_x = math.cos(x_closest) / 100
      control_y = math.sin(x_closest) / 100

      # control_x = math.cos(x_closest) / 50
      # control_y = math.sin(x_closest) / 50
      manipulability = panda_sim.get_manipulability()
      if (manipulability < 0.05):
        panda_sim.execute([-control_x, -control_y, 0, 15])
        panda_sim.execute([control_x * 10, control_y * 10, 0 ,2])
        panda_sim.step()
        # control_x *= 10
        # control_y *= 10
      panda_sim.execute([control_x, control_y, 0, 0.02])

      print(manipulability)
      # # if (manipulability < 0.01):
      # panda_sim.freeze_panda()        
      # #   panda_sim.bullet_client.stepSimulation()

if args.task == 4:
    utils.setup_dynamic_triangle_push(panda_sim)
    pdef = setup_pdef(panda_sim)
    object_pos = panda_sim.get_object_pos()
    eef_pos, _ = panda_sim.get_ee_pose()
    dataset = []
    for _ in range (10):
    # Get contact with the object
      while (not panda_sim.in_collision_with_object()):
        object_pos = panda_sim.get_object_pos()
        eef_pos, _ = panda_sim.get_ee_pose()
        control = normalize_vector(eef_pos, object_pos) / 100
        panda_sim.execute([control[0], control[1], 0, 1.2])

    # Generate a few random controls after getting contact to estimate the model
      object_pos = panda_sim.get_object_pos()
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = get_pos_angle(eef_pos, object_pos)
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
      # print("normal", oppo_pos_angle, "pos_angle", pos_angle)
      control_angle = random.uniform(-1.2, 1.2)
      # control_angle = -1
      # control_angle = random.uniform(-math.pi, math.pi)

      control_x = math.cos(control_angle + oppo_pos_angle) / 8000
      control_y = math.sin(control_angle + oppo_pos_angle) / 8000
      # control_x = math.cos(control_angle) / 200
      # control_y = math.sin(control_angle) / 200
      # print("execution", control_angle + oppo_pos_angle, "control angle", control_angle)
      panda_sim.execute([control_x, control_y, 0, 0.5])
      current_object_velocity, _ = panda_sim.get_object_velocity()
      while (math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2 + current_object_velocity[2] ** 2) > 0.0001):
        current_object_velocity, _ = panda_sim.get_object_velocity()
        panda_sim.execute([0, 0, 0, 1])
      end_object_pos = panda_sim.get_object_pos()
      object_vel_angle = math.atan2((end_object_pos[1] - object_pos[1]), (end_object_pos[0] - object_pos[0]))
      # print("result", object_vel_angle)
      object_vel_angle = utils.normalize_angle(object_vel_angle - oppo_pos_angle)
      dataset.append((control_angle, object_vel_angle))
      print(len(dataset))

    # data = np.array(dataset)
    data = np.load("20_data.npy")
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
        time.sleep(10000)
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

      control_x = math.cos(x_closest) / 100
      control_y = math.sin(x_closest) / 100

      # control_x = math.cos(x_closest) / 50
      # control_y = math.sin(x_closest) / 50
      manipulability = panda_sim.get_manipulability()
      if (manipulability < 0.05):
        panda_sim.execute([-control_x, -control_y, 0, 15])
        panda_sim.execute([control_x * 30, control_y * 30, 0 ,2])
        # control_x *= 10
        # control_y *= 10
      panda_sim.execute([control_x, control_y, 0, 0.02])

      print(manipulability)

if args.task==5:
  utils.setup_dynamic_cylinder_push(panda_sim, 0.002)
  pdef = setup_pdef(panda_sim)
  # Collect data by pushing mildly towards the center to avoid manipulability limits
  start = -0.8
  end = 0.8
  num_points = 20
  step = (end - start) / (num_points - 1)
  control_angles = [start + i * step for i in range(num_points)]

  dataset = []
  if args.pretraining:
    for _ in range (0, num_points):
      object_pos = panda_sim.get_object_pos()
      print("object position", object_pos)
      # origin = [0.1, 0.1]
      origin = [0.1, 0.1]
      # Assuming pushing towards [0.1, 0.1]
      pos_angle = math.atan2(object_pos[1]- origin[1], object_pos[0]- origin[0])
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
      coordinates = [object_pos[0] + 0.04 * math.cos(pos_angle), object_pos[1] + 0.04 * math.sin(pos_angle)]
      print("coordinates", coordinates)
      panda_sim.move(coordinates)
      control_angle = control_angles[_]
      # control_angle = random.uniform(-0.7, 0.7)
      print(control_angle)
      # control_angle = 0
      execution_angle = control_angle + oppo_pos_angle
      control_x = math.cos(execution_angle) / 35
      control_y = math.sin(execution_angle) / 35
      print("control", control_x, control_y)
      print("start executing")
      wpts, valid = panda_sim.execute([control_x, control_y, 0, 1])
      panda_sim.freeze_panda()
      # panda_sim.execute([0, 0, 0, 1])
      print("finish executing")
      current_object_velocity, _ = panda_sim.get_object_velocity()
      while (math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2) > 0.0001):
        current_object_velocity, _ = panda_sim.get_object_velocity()
        # print("Velocity", math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2))
        panda_sim.step()
      end_object_pos = panda_sim.get_object_pos()
      object_vel_angle = math.atan2((end_object_pos[1] - object_pos[1]), (end_object_pos[0] - object_pos[0]))
      object_vel_angle = utils.normalize_angle(object_vel_angle - oppo_pos_angle)
      displacement = math.sqrt((end_object_pos[0] - object_pos[0]) ** 2 + (end_object_pos[1] - object_pos[1]) ** 2)
      if (displacement > 0.01 and valid):
        dataset.append((control_angle, object_vel_angle))
      print(len(dataset))
    data = np.array(dataset)
    if args.save_data:
      np.save("dynamic_cylinder_push.npy", data)
  else:
    data = np.load("dynamic_cylinder_push.npy")
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

  degrees = [2, 3, 4, 5]
  polynomial_models = []
  scores = []

  for degree in degrees:
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(x.reshape(-1, 1), y)
    polynomial_models.append(poly_model)
    
    # Predict and evaluate
    y_pred = poly_model.predict(x.reshape(-1, 1))
    score = r2_score(y, y_pred)
    scores.append(score)
    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = linear_model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, color='red', label='Linear Fit')
    plt.legend()
    # plt.show()
  # best_model = polynomial_models[1]
  best_model = linear_model
  x_fit = np.linspace(x.min(), x.max(), 400)
  y_fit = best_model.predict(x_fit.reshape(-1, 1))
  plt.plot(x_fit, y_fit, color='red', label='Linear Fit')
  plt.legend()

  plt.show()

  object_pos = panda_sim.get_object_pos()
  original_object_pos = object_pos
  eef_pos, _ = panda_sim.get_ee_pose()

  desired_object_vel_angle = math.atan2(panda_sim.cylinder_push_goal[1] - object_pos[1], panda_sim.cylinder_push_goal[0] - object_pos[0])
  compare_angle = desired_object_vel_angle
  print("desird_object_vel_angle", desired_object_vel_angle)
  shift_angle = args.shift_angle
  pos_angle = utils.normalize_angle(desired_object_vel_angle + math.pi + shift_angle)
  # pos_angle = utils.normalize_angle(desired_object_vel_angle + math.pi)
  oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
  desired_object_vel_angle = utils.normalize_angle(desired_object_vel_angle - oppo_pos_angle)
  # desired_object_vel_angle = 0

  coordinates = [object_pos[0] + 0.04 * math.cos(pos_angle), object_pos[1] + 0.04 * math.sin(pos_angle)]
  panda_sim.move(coordinates)

  x_range = np.linspace(-0.8, 0.8, 1000)
  y_pred = best_model.predict(x_range.reshape(-1, 1))

  differences = np.abs(y_pred - desired_object_vel_angle)

  min_diff_x = np.argmin(differences)
  x_closest = x_range[min_diff_x]
  print("x_closest = ", x_closest)
  x_closest += oppo_pos_angle
  
  # hit directly in the direction of the goal
  # x_closest = oppo_pos_angle
  control_x = math.cos(x_closest) / 10
  control_y = math.sin(x_closest) / 10

  time.sleep(1)
  panda_sim.execute([control_x, control_y, 0, 1])
  panda_sim.execute([0, 0, 0, 5])
  
  current_object_velocity, _ = panda_sim.get_object_velocity()
  prev_object_pos = panda_sim.get_object_pos()
  while (math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2) > 0.0001):
      # utils.draw_line(panda_sim, prev_object_pos, object_pos, c=[0, 0, 1], w=2)
      object_pos = panda_sim.get_object_pos()
      prev_object_pos = panda_sim.get_object_pos()
      current_object_velocity, _ = panda_sim.get_object_velocity()
      
      # print("Velocity", math.sqrt(current_object_velocity[0] ** 2 + current_object_velocity[1] ** 2))
      panda_sim.step()
  end_pos = panda_sim.get_object_pos()
  print("end_pos", end_pos)
  gamma = math.atan2(end_pos[1] - original_object_pos[1], end_pos[0] - original_object_pos[0])
  print("gamma", math.atan2(end_pos[1] - original_object_pos[1], end_pos[0] - original_object_pos[0]))
  print("angle difference", abs(gamma - compare_angle))
  time.sleep(10000)

if args.task==6:
  utils.setup_cylinder_push(panda_sim,)
  utils.setup_cube_push(panda_sim)
  utils.setup_triangle_push(panda_sim)
  time.sleep(10000)

if args.task == 7:
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
    start = -0.8
    end = 0.8
    num_points = 5
    step = (end - start) / (num_points - 1)
    control_angles = [start + i * step for i in range(num_points)]
    for i in range (num_points):
      object_pos = panda_sim.get_object_pos()
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = get_pos_angle(eef_pos, object_pos)
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)
      # print("normal", oppo_pos_angle, "pos_angle", pos_angle)
  
      # control_angle = random.uniform(-0.8, 0.8)
      # control_angle = -1
      # control_angle = random.uniform(-math.pi, math.pi)
      control_angle = control_angles[i]
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
    np.save("cylinder_5.npy", data)
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

    trajectory = utils.generate_circular_trajectory([0, 0], 0.1)
    # trajectory = [[0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
    # trajectory = [
    #   [0.1, 0.0],
    #   [0.1, 0.02],
    #   [0.1, 0.04],
    #   [0.1, 0.06],
    #   [0.1, 0.08],
    #   [0.1, 0.08],
    #   [0.08, 0.08],
    #   [0.06, 0.08],
    #   [0.04, 0.08],
    #   [0.02, 0.08],
    #   [0.02, 0.08],
    #   [0.02, 0.06],
    #   [0.02, 0.04],
    #   [0.02, 0.02],
    #   [0.02, 0.0],
    #   [0.02, 0.0],
    #   [0.04, 0.0],
    #   [0.06, 0.0],
    #   [0.08, 0.0],
    #   [0.1, 0.0]
    # ] 
    current_goal_index = 0
    prev_object_pos = panda_sim.get_object_pos()
    start_drawing = False
    object_traj = []
    while True:
      object_pos = panda_sim.get_object_pos()
      
      if (start_drawing):
        utils.draw_line(panda_sim, prev_object_pos, object_pos, c=[0, 0, 1], w=2)
        object_traj.append(object_pos)
      prev_object_pos = object_pos
      #Check if the goal state is achieved
      if (utils.distance(object_pos, trajectory[current_goal_index]) < 0.02):
        print("Goal reached!!!", current_goal_index)
        current_goal_index += 1
        current_goal_index = current_goal_index % len(trajectory)
        if (current_goal_index == 5 and start_drawing):
          object_trace = np.array(object_traj)
          np.save("cylinder_traj.npy", object_trace)
        if (current_goal_index == 5):
          start_drawing = True
        # time.sleep(10000)
      eef_pos, _ = panda_sim.get_ee_pose()
      pos_angle = math.atan2((eef_pos[1] - object_pos[1]), (eef_pos[0] - object_pos[0]))
      oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

      desired_object_vel_angle = math.atan2(trajectory[current_goal_index][1] - object_pos[1], trajectory[current_goal_index][0] - object_pos[0])
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
    

if args.task == 8:
  # utils.setup_triangle_push(panda_sim)
  utils.setup_cube_push(panda_sim)
  pdef = setup_pdef(panda_sim)
  object_pos = panda_sim.get_object_pos()
  eef_pos, _ = panda_sim.get_ee_pose()

  # Get contact with the object
  while (not panda_sim.in_collision_with_object()):
    control = normalize_vector(eef_pos, object_pos) / 100
    panda_sim.execute([control[0], control[1], 0 , 1])
  print("Contact detected")

  data = np.load("cylinder_5.npy")
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlim([-math.pi, math.pi])
  ax.set_ylim([-math.pi, math.pi])

  object_vel_angles = data[:, 1]
  control_angles = data[:, 0]

  ax.scatter(control_angles, object_vel_angles)
  ax.set_xlabel('Control Angle')
  ax.set_ylabel('Object Velocity Angle')
  
  # plt.show()

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

  trajectory = utils.generate_circular_trajectory([0, 0], 0.1)
  # trajectory = [
  #   [0.1, 0.0],
  #   [0.1, 0.02],
  #   [0.1, 0.04],
  #   [0.1, 0.06],
  #   [0.1, 0.08],
  #   [0.1, 0.08],
  #   [0.08, 0.08],
  #   [0.06, 0.08],
  #   [0.04, 0.08],
  #   [0.02, 0.08],
  #   [0.02, 0.08],
  #   [0.02, 0.06],
  #   [0.02, 0.04],
  #   [0.02, 0.02],
  #   [0.02, 0.0],
  #   [0.02, 0.0],
  #   [0.04, 0.0],
  #   [0.06, 0.0],
  #   [0.08, 0.0],
  #   [0.1, 0.0]
  # ] 
  # trajectory = [[0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
  current_goal_index = 0
  prev_object_pos = panda_sim.get_object_pos()
  start_drawing = False
  object_traj = []
  while True:
    object_pos = panda_sim.get_object_pos()
    if (start_drawing):
      utils.draw_line(panda_sim, prev_object_pos, object_pos, c=[0, 0, 1], w=2)
      object_traj.append(object_pos)
    #Check if the goal state is achieved
    prev_object_pos = object_pos
    if (utils.distance(object_pos, trajectory[current_goal_index]) < 0.01):
      print("Goal reached!!!", current_goal_index)
      current_goal_index += 1
      current_goal_index = current_goal_index % len(trajectory)
      if (current_goal_index == 5 and start_drawing):
          object_trace = np.array(object_traj)
          np.save("cube_traj.npy", object_trace)
      if (current_goal_index == 5):
        start_drawing = True

    eef_pos, _ = panda_sim.get_ee_pose()
    pos_angle = math.atan2((eef_pos[1] - object_pos[1]), (eef_pos[0] - object_pos[0]))
    oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

    desired_object_vel_angle = math.atan2(trajectory[current_goal_index][1] - object_pos[1], trajectory[current_goal_index][0] - object_pos[0])
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

if args.task == 9:
  utils.setup_triangle_push(panda_sim)
  # utils.setup_cube_push(panda_sim)
  pdef = setup_pdef(panda_sim)
  object_pos = panda_sim.get_object_pos()
  eef_pos, _ = panda_sim.get_ee_pose()

  # Get contact with the object
  while (not panda_sim.in_collision_with_object()):
    control = normalize_vector(eef_pos, object_pos) / 100
    panda_sim.execute([control[0], control[1], 0 , 1])
  print("Contact detected")

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
  
  # plt.show()

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

  trajectory = utils.generate_circular_trajectory([0, 0], 0.1)
  # trajectory = [
  #   [0.1, 0.0],
  #   [0.1, 0.02],
  #   [0.1, 0.04],
  #   [0.1, 0.06],
  #   [0.1, 0.08],
  #   [0.1, 0.08],
  #   [0.08, 0.08],
  #   [0.06, 0.08],
  #   [0.04, 0.08],
  #   [0.02, 0.08],
  #   [0.02, 0.08],
  #   [0.02, 0.06],
  #   [0.02, 0.04],
  #   [0.02, 0.02],
  #   [0.02, 0.0],
  #   [0.02, 0.0],
  #   [0.04, 0.0],
  #   [0.06, 0.0],
  #   [0.08, 0.0],
  #   [0.1, 0.0]
  # ] 
  current_goal_index = 0
  prev_object_pos = panda_sim.get_object_pos()
  start_drawing = False
  object_traj = []

  while True:
    object_pos = panda_sim.get_object_pos()
    if (start_drawing):
      utils.draw_line(panda_sim, prev_object_pos, object_pos, c=[0, 0, 1], w=2)
      object_traj.append(object_pos)
    #Check if the goal state is achieved
    prev_object_pos = object_pos
    if (utils.distance(object_pos, trajectory[current_goal_index]) < 0.03):
      print("Goal reached!!!", current_goal_index)
      current_goal_index += 1
      current_goal_index = current_goal_index % len(trajectory)
      if (current_goal_index == 5 and start_drawing):
          object_trace = np.array(object_traj)
          np.save("triangle_traj.npy", object_trace)
      if (current_goal_index == 5):
        start_drawing = True
    else:
      print("distance to goal", utils.distance(object_pos, panda_sim.cylinder_push_goal))
    eef_pos, _ = panda_sim.get_ee_pose()
    pos_angle = math.atan2((eef_pos[1] - object_pos[1]), (eef_pos[0] - object_pos[0]))
    oppo_pos_angle = utils.normalize_angle(pos_angle + math.pi)

    desired_object_vel_angle = math.atan2(trajectory[current_goal_index][1] - object_pos[1], trajectory[current_goal_index][0] - object_pos[0])
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
