"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import sys

import cvxpy
import matplotlib.pyplot as plt
import numpy as np

import reachability.main as reach
import reachset_transform.main as transform
import Car_dimensions.main as car_reach
import pypoman

import multiprocessing as mp

import timeit
import random
from matplotlib.patches import Rectangle

from shapely.geometry import Polygon as shapely_poly
from pytope import Polytope as py_poly

from mpire import WorkerPool

import threading

sys.path.append("../../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1])  # input difference cost matrix
Q = np.diag([0.01, 0.01, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.4  # [s] time tick

# Vehicle parameters
LENGTH = 2.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 0.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 5  # 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

models = reach.get_models()
theta_min_model = reach.get_theta_min_model()
theta_max_model = reach.get_theta_max_model()


class State:
    """
   vehicle state class
   """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, a, delta):
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    x_error = random.uniform(-0.1, 0.1)
    y_error = random.uniform(-0.1, 0.1)
    yaw_error = random.uniform(-0.1, 0.1)
    state.x = state.x + state.v * math.cos(state.yaw) * DT + x_error * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT + y_error * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT + yaw_error * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
   MPC contorl with updating operational point iteraitvely
   """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
   linear mpc control

   xref: reference point
   xbar: operational point
   x0: initial state
   dref: reference steer angle
   """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
   Simulation

   cx: course x position list
   cy: course y position list
   cy: course yaw position list
   ck: course curvature list
   sp: speed profile
   dl: course tick [m]

   """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0

    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    safe = True
    while MAX_TIME >= time:

        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        # print("steering angle: ", odelta)
        # print("acceleration: ", oa)
        # print("velocity: ", state.v)

        state = update_state(state, ai, di)  # di is steering angle (delta)
        time = time + DT

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            # plot reachset

            # safe = reachability(oa, odelta, state)


            # plot obstacles
            # specify the location of (left,bottom),width,height

            car_xs = [state.x, state.x + 2.5, state.x + 2.5, state.x]
            car_ys = [state.y - 1, state.y - 1, state.y + 1, state.y + 1]
            rot_car_xs, rot_car_ys = transform.rotate(car_xs, car_ys, state.yaw, state.x, state.y)
            rot_car_coords = transform.get_coords(rot_car_xs, rot_car_ys)
            car_box = shapely_poly(rot_car_coords)
            plt.fill(*obstacle.exterior.xy)
            if obstacle.intersects(car_box):
                safe = False
                print("car crashes into obstacle")
                break

            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))

            plt.pause(0.0001)
            if not safe:
                print("reachable set intersects with obstacle")
                break
            elif obstacle.intersects(car_box):
                print("car crashes into obstacle")
                break

    plt.close("all")
    return safe


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_left_turn_course(dl):
    ax = [0, 10, 20, 21.0, 30.0, 35, 36, 36, 36]
    ay = [0, 0, 0, 0, 1.0, 5.0, 10, 15, 20]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_right_turn_course(dl):
    ax = [0, 10, 20, 21.0, 30.0, 35, 36, 36, 36]
    ay = [0, 0, 0, 0, -1.0, -5.0, -10, -15, -20]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_uturn_course(dl):
    ax = [0, 10, 20, 30, 35, 36, 35, 30, 20, 10]
    ay = [0, 0, 0, 1, 3, 5, 7, 9, 10, 10]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_change_lane_course(dl):
    ax = [0, 10, 20, 30, 40, 50, 60]
    ay = [0, 0, 0, 2, 4, 4, 4]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def reachability(oa, odelta, state):
    safe = True
    parallel_start = timeit.default_timer()

    input_list = []
    for i in range(100):
        nn_input = [i + 1, oa[0], odelta[0], oa[1], odelta[1], oa[2], odelta[2], oa[3], odelta[3], oa[4],
                    odelta[4],
                    state.v]
        input_list.append(nn_input)
    gpu_input_list = reach.gpu_inputs_list(input_list)
    sf_list = reach.get_reachset(gpu_input_list, models)
    theta_min_list = reach.get_theta_min_list(gpu_input_list, theta_min_model)
    theta_max_list = reach.get_theta_max_list(gpu_input_list, theta_max_model)

    for reach_iter in range(100):
        new_sf_list = []
        for dirs in range(len(sf_list)):
            new_sf_list.append(sf_list[dirs][0][reach_iter][0])
        poly_reach = reach.sf_to_poly(new_sf_list)

        full_reach_poly = car_reach.add_car_to_reachset(poly_reach, theta_min_list[0][reach_iter][0],
                                                        theta_max_list[0][reach_iter][0])
        final_reach_poly = transform.transform_poly(full_reach_poly, state.yaw, state.x, state.y)

        reachpoly = shapely_poly(final_reach_poly.V)
        if obstacle.intersects(reachpoly):
            safe = False

        # print(obstacle.intersects(reachpoly))
        # final_reach_poly.plot()

    parallel_stop = timeit.default_timer()
    print('parallel Time Taken: ', (parallel_stop - parallel_start))

    return safe


def main(course_num, parallel_iter):
    random.seed(parallel_iter)
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    courses = [get_straight_course(dl), get_left_turn_course(dl), get_right_turn_course(dl), get_uturn_course(dl),
               get_change_lane_course(dl)]

    cx, cy, cyaw, ck = courses[course_num]

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    safe = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)

    plt.show()

    return safe


def get_obstacle(obstacle_num):
    obstacles = []
    straight_obs = shapely_poly([(10, 1.5), (50, 1.5), (50, 5), (10, 5)])  # semi-done
    obstacles.append(straight_obs)
    left_obs = shapely_poly([(37, -10), (39, -10), (39, 20), (37, 20)])  # semi-done
    obstacles.append(left_obs)
    right_obs = shapely_poly([(37, 10), (39, 10), (39, -20), (37, -20)])  # semi-done
    obstacles.append(right_obs)
    uturn_obst = shapely_poly([(10, 2.5), (30, 2.5), (33, 4), (33, 6), (30, 7.5), (10, 7.5)])  # done
    obstacles.append(uturn_obst)
    change_lane_obst = shapely_poly([(34.5, 0), (60, 0), (60, 2), (34.5, 2)])  # done
    obstacles.append(change_lane_obst)

    return obstacles[obstacle_num]


if __name__ == '__main__':
    benchmark = 4
    crashes = 0
    obstacle = get_obstacle(benchmark)
    # for i in range(100):
    #     random.seed(i)
    #     safe_run = main(benchmark, i)
    #     if not safe_run:
    #         crashes += 1

    with WorkerPool(n_jobs=8, shared_objects=benchmark) as pool:
        safety_list = pool.map(main, range(100))
    print("number of crashes: ", 100-sum(safety_list))
