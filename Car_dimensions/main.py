# import os, sys
#
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

# import matplotlib.pyplot as plt
# import pypoman

import skgeom as sg
from skgeom import minkowski
import reachset_transform.main as transform
import numpy as np
from pytope import Polytope

car_V = np.array([[0.0, -1.0], [2.5, -1.0], [2.5, 1.0], [0.0, 1.0]])
car_poly = Polytope(car_V)

car_xs = [0.0, 2.5, 2.5, 0.0]
car_ys = [-1.0, -1.0, 1.0, 1.0]


def car_points(theta_min, theta_max):
    theta = (theta_min + theta_max) / 2
    delta_theta = theta_max - theta  # or theta - theta_min
    new_car_poly = uncertain_theta_effect(delta_theta)
    rot_car_poly = transform.poly_rotate(new_car_poly, theta)
    return rot_car_poly


def uncertain_theta_effect(delta_theta):
    rot_xs, rot_ys = transform.rotate(car_xs, car_ys, delta_theta)
    neg_rot_xs, neg_rot_ys = transform.rotate(car_xs, car_ys, -delta_theta)

    new_car_V = np.array([[min(neg_rot_xs), min(neg_rot_ys)], [max(rot_xs), min(neg_rot_ys)], [max(rot_xs), max(rot_ys)], [min(neg_rot_xs), max(rot_ys)]])
    new_car_poly = Polytope(new_car_V)

    return new_car_poly


def minkowski_sum(p1, p2):

    result = p1 + p2
    return result


def add_car_to_reachset(poly_reach, theta_min, theta_max):
    final_car_poly = car_points(theta_min, theta_max)
    full_reach_poly = minkowski_sum(poly_reach, final_car_poly)

    return full_reach_poly

# if __name__ == '__main__':
    # plt.figure()
    # coords = []
    # for xs, ys in zip(car_xs, car_ys):
    #     coords.append([xs, ys])
    #
    # pypoman.polygon.plot_polygon(coords, alpha=0.5)
    #
    # new_xs, new_ys = transform.rotate(car_xs, car_ys, 0.15)
    # coords = []
    # for xs, ys in zip(new_xs, new_ys):
    #     coords.append([xs, ys])
    # pypoman.polygon.plot_polygon(coords, alpha=0.2)

    # new_xs, new_ys = uncertain_theta_effect(0.15)
    # coords = []
    # for xs, ys in zip(new_xs, new_ys):
    #     coords.append([xs, ys])
    # pypoman.polygon.plot_polygon(coords, alpha=0.4, color='r')
    #
    # rot_xs, rot_ys = transform.rotate(new_xs, new_ys, 0.4)
    # coords = []
    # for xs, ys in zip(rot_xs, rot_ys):
    #     coords.append([xs, ys])
    # pypoman.polygon.plot_polygon(coords, alpha=0.1, color='r')
    #
    #
    # plt.show()
