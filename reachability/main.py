#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import numpy
import torch

import pypoman
import reachset_transform.main as transform

import matplotlib.pyplot as plt

# CUDA for PyTorch
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
car_xs = [0, 2.5, 2.5, 0]
car_ys = [-1, -1, 1, 1]


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    # no retraining
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval().to(device)

    return model


def get_models():
    models = []
    for i in range(8):
        file = 'reachability/bicyclemodels/new_bicycle_dir' + str(i + 1) + '_100kSamples_100kEpochs.pth'
        nn = load_checkpoint(file)
        models.append(nn)
    return models


def get_reachset(nn_input, models):
    sf = []
    # test data for initial region as shown in JuliaReach documentation
    x = torch.tensor(nn_input).to(device)
    models_optim = models

    x_mean = 4.5028
    x_std = 16.2052

    x = (x - x_mean) / x_std

    y_std = [2.5071, 1.1776, 1.9493, 1.7887, 2.0338, 1.1589, 2.4328, 2.0179]
    y_mean = [2.6990, 0.2936, -1.8478, -2.0807, -1.8901, 0.2159, 2.5820, 2.4274]

    for i in range(8):
        rnn_forward = models_optim[i][0].forward(x.float().view(1, 1, 12))
        val = models_optim[i][1].forward(rnn_forward[0]).cpu()
        val = (val * y_std[i]) + y_mean[i]
        val = float(val) + 0.02
        sf.append(val)

    return sf


def sf_to_ver(sf):
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


def get_theta_min(nn_input, model_optim):
    # test data for initial region as shown in JuliaReach documentation
    x = torch.tensor(nn_input).to(device)

    x_mean = 4.5028
    x_std = 16.2052

    x = (x - x_mean) / x_std

    y_std = 0.4052
    y_mean = -0.0478

    rnn_forward = model_optim[0].forward(x.float().view(1, 1, 12))
    val = model_optim[1].forward(rnn_forward[0]).cpu()
    val = (val * y_std) + y_mean
    val = float(val)

    return val


def get_theta_max(nn_input, model_optim):
    # test data for initial region as shown in JuliaReach documentation
    x = torch.tensor(nn_input).to(device)

    x_mean = 4.5028
    x_std = 16.2052

    x = (x - x_mean) / x_std

    y_std = 0.4274
    y_mean = 0.1595

    rnn_forward = model_optim[0].forward(x.float().view(1, 1, 12))
    val = model_optim[1].forward(rnn_forward[0]).cpu()
    val = (val * y_std) + y_mean
    val = float(val)

    return val


def get_theta_min_model():
    file = 'reachability/bicyclemodels/new_bicycle_thetaMin_100kSamples_100kEpochs.pth'
    model_optim = load_checkpoint(file)
    return model_optim


def get_theta_max_model():
    file = 'reachability/bicyclemodels/new_bicycle_thetaMax_100kSamples_100kEpochs.pth'
    model_optim = load_checkpoint(file)
    return model_optim


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
