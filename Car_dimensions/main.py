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



car_xs = [0.0, 2.5, 2.5, 0.0]
car_ys = [-1.0, -1.0, 1.0, 1.0]


def car_points(theta_min, theta_max):
    theta = (theta_min + theta_max) / 2
    delta_theta = theta_max - theta  # or theta - theta_min
    new_car_xs, new_car_ys = uncertain_theta_effect(delta_theta)
    rot_car_xs, rot_car_ys = transform.rotate(new_car_xs, new_car_ys, theta)
    return rot_car_xs, rot_car_ys


def uncertain_theta_effect(delta_theta):
    rot_xs, rot_ys = transform.rotate(car_xs, car_ys, delta_theta)
    neg_rot_xs, neg_rot_ys = transform.rotate(car_xs, car_ys, -delta_theta)
    new_car_xs = [min(neg_rot_xs), max(rot_xs), max(rot_xs), min(neg_rot_xs)]
    new_car_ys = [min(neg_rot_ys), min(neg_rot_ys), max(rot_ys), max(rot_ys)]

    return new_car_xs, new_car_ys


def minkowski_sum(xs1, ys1, xs2, ys2):
    p1_array = []
    p2_array = []
    for x, y in zip(xs1, ys1):
        p1_array.append(sg.Point2(x, y))
    p1 = sg.Polygon(p1_array)
    for x, y in zip(xs2, ys2):
        p2_array.append(sg.Point2(x, y))
    p2 = sg.Polygon(p2_array)

    result = minkowski.minkowski_sum(p1, p2)
    return result.outer_boundary().coords


def add_car_to_reachset(xs_reach, ys_reach, theta_min, theta_max):
    car_xs, car_ys = car_points(theta_min, theta_max)
    full_reach_coords = minkowski_sum(xs_reach, ys_reach, car_xs, car_ys)

    return full_reach_coords

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
