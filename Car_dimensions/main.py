"""

Incorporating car dimensions into computed reachable sets

author: Abdelrahman Hekal

"""
import reachset_transform.main as transform
import numpy as np
from pytope import Polytope

car_V = np.array([[0.0, -1.0], [2.5, -1.0], [2.5, 1.0], [0.0, 1.0]])
car_poly = Polytope(car_V)

car_xs = [0.0, 2.5, 2.5, 0.0]
car_ys = [-1.0, -1.0, 1.0, 1.0]


def car_points(theta_min, theta_max):
    """
   Add effect of uncertainty in orientation by enlarging box that represents car.
   Then rotate car to current orientation
   """
    theta = (theta_min + theta_max) / 2
    delta_theta = theta_max - theta  # or theta - theta_min
    new_car_poly = uncertain_theta_effect(delta_theta)
    rot_car_poly = transform.poly_rotate(new_car_poly, theta)
    return rot_car_poly


def uncertain_theta_effect(delta_theta):
    """
   Add effect of uncertainty in orientation by enlarging box that represents car.
   """
    rot_xs, rot_ys = transform.rotate(car_xs, car_ys, delta_theta)
    neg_rot_xs, neg_rot_ys = transform.rotate(car_xs, car_ys, -delta_theta)

    new_car_V = np.array([[min(neg_rot_xs), min(neg_rot_ys)], [max(rot_xs), min(neg_rot_ys)], [max(rot_xs), max(rot_ys)], [min(neg_rot_xs), max(rot_ys)]])
    new_car_poly = Polytope(new_car_V)

    return new_car_poly


def minkowski_sum(p1, p2):

    result = p1 + p2
    return result


def add_car_to_reachset(poly_reach, theta_min, theta_max):
    """
   Add dimensions of vehicle to computed reachset for midpoint of rear-axle
   """
    final_car_poly = car_points(theta_min, theta_max)
    full_reach_poly = minkowski_sum(poly_reach, final_car_poly)

    return full_reach_poly
