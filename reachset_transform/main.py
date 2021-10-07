import matplotlib.pyplot as plt
import math
import numpy as np


class Transform:

    def transform_coords(self, coords, angle, xd, yd):
        xs, ys = self.get_list(coords)
        xs_tr, ys_tr = self.transform(xs, ys, angle, xd, yd)
        coords = self.get_coords(xs_tr, ys_tr)
        return coords

    def get_coords(self, xs, ys):
        coords = []
        for x, y in zip(xs, ys):
            coord = [x, y]
            coords.append(coord)
        return coords

    def get_list(self, coords):
        xs = []
        ys = []
        for coord in coords:
            xs.append(coord[0])
            ys.append(coord[1])
        return xs, ys

    def transform(self, xs, ys, angle, xd, yd):
        xs_rot, ys_rot = self.rotate(xs, ys, angle)
        xs_trans, ys_trans = self.translate(xs_rot, ys_rot, xd, yd)
        #self.plot_oct(xs_trans, ys_trans)
        return xs_trans, ys_trans

    def transform_file(self, name, angle, xd, yd, colour='g'):
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
                    #self.plot_oct(xs,ys, colour)
                    xs_trans, ys_trans = self.transform(xs, ys, angle, xd, yd)
                    self.plot_oct(xs_trans, ys_trans, colour)
                xs = []
                ys = []

    def plot_oct(self, xs, ys, colour = 'g'):
        coord = [[xs[0], ys[0]], [xs[1], ys[1]], [xs[2], ys[2]], [xs[3], ys[3]],
                 [xs[4], ys[4]], [xs[5], ys[5]], [xs[6], ys[6]], [xs[7], ys[7]], [xs[8], ys[8]]]

        xs, ys = zip(*coord)  # create lists of x and y values

        plt.plot(xs, ys, colour)

    # Press the green button in the gutter to run the script.

    def rotate(self, xs, ys, angle):
        cx = 0
        cy = 0
        xs = np.array(xs)
        ys = np.array(ys)
        xs -= cx
        ys -= cy
        xs_new = xs * math.cos(angle) - ys * math.sin(angle)
        ys_new = ys * math.cos(angle) + xs * math.sin(angle)
        xs = xs_new + cx
        ys = ys_new + cy

        return xs, ys

    def translate(self, xs, ys, xd, yd):  # xd, yd are differnce in xs and ys required for translation
        xs = np.array(xs)
        ys = np.array(ys)
        xs += xd
        ys += yd

        return xs, ys

#test for rotation
if __name__ == '__main__':
    # proof of concept that rotation and translation work. difference in two sets is because octagon directions are set and not rotated
    plt.figure()
    transform = Transform()
    transform.transform_file(name="data", angle=0.8, xd=0, yd=0)
    transform.transform_file(name="data_theta_1.57", angle=0, xd=0, yd=0, colour='r')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
