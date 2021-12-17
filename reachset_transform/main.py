"""

Transform reachable sets to account for position and orientation.
This is achieve by rotation and translation of the reachable sets.

author: Abdelrahman Hekal

"""
import matplotlib.pyplot as plt
import math
import numpy as np
from pytope import Polytope


def get_coords(xs, ys):
    """
   Get a list of coordinates from lists for x-points and y-points
   """
    coords = []
    for x, y in zip(xs, ys):
        coord = [x, y]
        coords.append(coord)
    return coords


def transform_coords(coords, angle, xd, yd):
    """
   rotate and translate coords by desired angle and position
   """
    xs, ys = get_list(coords)
    xs_tr, ys_tr = transform(xs, ys, angle, xd, yd)
    coords = get_coords(xs_tr, ys_tr)
    return coords


def get_list(coords):
    """
   Get lists for x-points and y-points from a list of coordinates
   """
    xs = []
    ys = []
    for coord in coords:
        xs.append(coord[0])
        ys.append(coord[1])
    return xs, ys


def transform(xs, ys, angle, xd, yd):
    """
   rotate and translate lists of x-points and y-points by desired angle and position
   """
    xs_rot, ys_rot = rotate(xs, ys, angle)
    xs_trans, ys_trans = translate(xs_rot, ys_rot, xd, yd)
    # self.plot_oct(xs_trans, ys_trans)
    return xs_trans, ys_trans


def transform_poly(poly, angle, xd, yd):
    """
   rotate and translate a polygon by desired angle and position
   """
    rot_poly = poly_rotate(poly, angle)
    trans_poly = translate_poly(rot_poly, xd, yd)

    return trans_poly


def transform_file(name, angle, xd, yd):
    """
   rotate and translate a file that contains vertices information outputted by Flow* by desired angle and position
   """
    f = open(name, "r")
    xs = []
    ys = []
    for line in f:
        if len(line) > 1:
            coord = line.split(" ")
            x = float(coord[0])
            y = float(coord[1])
            xs.append(x)
            ys.append(y)
        else:
            if len(xs) == 9:
                transform(xs, ys, angle, xd, yd)
            xs = []
            ys = []


def plot_oct(xs, ys):
    coord = [[xs[0], ys[0]], [xs[1], ys[1]], [xs[2], ys[2]], [xs[3], ys[3]],
             [xs[4], ys[4]], [xs[5], ys[5]], [xs[6], ys[6]], [xs[7], ys[7]], [xs[8], ys[8]]]

    xs, ys = zip(*coord)  # create lists of x and y values

    plt.plot(xs, ys, 'g')


# Press the green button in the gutter to run the script.

def rotate(xs, ys, angle, cx=0, cy=0):
    xs = np.array(xs)
    ys = np.array(ys)
    xs -= cx
    ys -= cy
    xs_new = xs * math.cos(angle) - ys * math.sin(angle)
    ys_new = ys * math.cos(angle) + xs * math.sin(angle)
    xs = xs_new + cx
    ys = ys_new + cy

    return xs, ys


def poly_rotate(poly, angle):
    rot = angle
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])
    P = rot_mat * poly

    return P


def translate(xs, ys, xd, yd):  # xd, yd are differnce in xs and ys required for translation
    xs = np.array(xs)
    ys = np.array(ys)
    xs += xd
    ys += yd

    return xs, ys


def translate_poly(poly, xd, yd):
    pd = [xd, yd]
    trans_poly = poly + pd

    return trans_poly
