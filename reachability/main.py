#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import numpy
import torch

import pypoman
from reachset_transform.main import Transform

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
        self.get_models()

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
        # test data for initial region as shown in JuliaReach documentation
        x = torch.tensor(nn_input).to(device)
        models_optim = self.models

        x_mean = 2.7558
        x_std = 8.3891

        x = (x - x_mean) / x_std

        y_std = [3.6415, 2.4452, 3.3113, 2.3879, 3.3268, 2.4325, 3.5830, 2.5383]
        y_mean = [3.8069, 0.4981, -2.5262, -2.9856, -3.0260, -0.0044, 3.3033, 3.3437]

        for i in range(8):
            rnn_forward = models_optim[i][0].forward(x.float().view(1, 1, 12))
            val = models_optim[i][1].forward(rnn_forward[0]).cpu()
            val = (val * y_std[i]) + y_mean[i]
            val = float(val)
            self.sf.append(val)

    def sf_to_ver_and_plot(self):
        sf = self.sf
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
        vertices = pypoman.compute_polytope_vertices(A, b)
        #print(len(vertices))
        #pypoman.polygon.plot_polygon(vertices)
        return vertices


if __name__ == '__main__':
    plt.figure()
    for i in range(50):
        reach = Reach()
        reach.get_reachset([i+1, 1.1, 0.1, 4.2, -0.5, 7.3, -0.5, 9.4, -0.5, 2.5, 0.1, 0.0])
        vertix = reach.sf_to_ver_and_plot()
        if len(vertix) != 0:
            pypoman.polygon.plot_polygon(vertix)
            transform = Transform()
            coords = transform.transform_coords(vertix, 3.14, 0.0, 0.0)
            pypoman.polygon.plot_polygon(coords)
        else:
            print("failed")
    plt.show()
