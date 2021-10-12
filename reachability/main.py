#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import numpy
import torch
import math

import skgeom as sg
from skgeom import minkowski
from skgeom.draw import draw

import pypoman
import reachset_transform.main as transform

import matplotlib.pyplot as plt

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


class Reach:

    def __init__(self):
        self.models = []
        self.get_models()
        self.sf = []
        self.car_xs = [0, 2.5, 2.5, 0]
        self.car_ys = [-1, -1, 1, 1]

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        # no retraining
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval().to(device)

        return model

    def get_models(self):
        for i in range(8):
            file = 'reachability/bicyclemodels/new_bicycle_dir' + str(i + 1) + '_100kSamples_100kEpochs.pth'
            nn = self.load_checkpoint(file)
            self.models.append(nn)

    def get_reachset(self, nn_input):
        self.sf = []
        # test data for initial region as shown in JuliaReach documentation
        x = torch.tensor(nn_input).to(device)
        models_optim = self.models

        x_mean = 4.5246
        x_std = 16.2002

        x = (x - x_mean) / x_std

        y_std = [2.3634, 1.1607, 2.1592, 1.9090, 2.2067, 1.1526, 2.3132, 1.9778]
        y_mean = [2.6133, 0.1772, -2.0733, -2.2305, -2.1561, 0.0864, 2.5138, 2.4472]

        for i in range(8):
            rnn_forward = models_optim[i][0].forward(x.float().view(1, 1, 12))
            val = models_optim[i][1].forward(rnn_forward[0]).cpu()
            val = (val * y_std[i]) + y_mean[i]
            val = float(val)
            self.sf.append(val)

    def sf_to_ver(self):
        sf = self.sf
        print(sf)
        A = numpy.array([
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
            [1, 0]])

        b = numpy.array(sf)
        vertices = pypoman.compute_polygon_hull(A, b)
        # print(len(vertices))
        # pypoman.polygon.plot_polygon(vertices)
        return vertices

    def get_theta_min(self, nn_input):
        # test data for initial region as shown in JuliaReach documentation
        x = torch.tensor(nn_input).to(device)
        file = 'reachability/bicyclemodels/new_bicycle_thetaMin_100kSamples_100kEpochs.pth'
        model_optim = self.load_checkpoint(file)

        x_mean = 4.5246
        x_std = 16.2002

        x = (x - x_mean) / x_std

        y_std = 0.4077
        y_mean = 0.0330

        rnn_forward = model_optim[0].forward(x.float().view(1, 1, 12))
        val = model_optim[1].forward(rnn_forward[0]).cpu()
        val = (val * y_std) + y_mean
        val = float(val)

        return val

    def get_theta_max(self, nn_input):
        # test data for initial region as shown in JuliaReach documentation
        x = torch.tensor(nn_input).to(device)
        file = 'reachability/bicyclemodels/new_bicycle_thetaMax_100kSamples_100kEpochs.pth'
        model_optim = self.load_checkpoint(file)

        x_mean = 4.5246
        x_std = 16.2002

        x = (x - x_mean) / x_std

        y_std = 0.4139
        y_mean = 0.1002

        rnn_forward = model_optim[0].forward(x.float().view(1, 1, 12))
        val = model_optim[1].forward(rnn_forward[0]).cpu()
        val = (val * y_std) + y_mean
        val = float(val)
        self.sf.append(val)

        return val




if __name__ == '__main__':
    plt.figure()
    for i in range(50):
        reach = Reach()
        reach.get_reachset([i + 1, 1.1, 0.1, 4.2, -0.5, 7.3, -0.5, 9.4, -0.5, 2.5, 0.1, 0.0])
        vertix = reach.sf_to_ver()
        if len(vertix) != 0:
            pypoman.polygon.plot_polygon(vertix)
            coords = transform.transform_coords(vertix, 3.14, 0.0, 0.0)
            pypoman.polygon.plot_polygon(coords)
        else:
            print("failed")
    plt.show()
