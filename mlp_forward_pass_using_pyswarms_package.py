# code by William Russell

# install pyswarms on device

import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

def PSO_forward_pass(W):

  X_pso = np.array([[0.05258131, 0.44914306]])

  # first layer
  net_h1 = (X_pso[0][0] * W[0]) + (X_pso[0][1] * W[1]) + 0.35
  out_h1 = 1/(1 + np.exp(-net_h1))

  net_h2 = (X_pso[0][0] * W[2]) + (X_pso[0][1] * W[3]) + 0.35
  out_h2 = 1/(1 + np.exp(-net_h2))

  # second layer
  net_o1 = (out_h1 * W[4]) + (out_h2 * W[5]) + 0.6
  out_o1 = 1/(1 + np.exp(-net_o1))

  net_o2 = (out_h1 * W[6]) + (out_h2 * W[7]) + 0.6
  out_o2 = 1/(1 + np.exp(-net_o2))

  # Total Error
  error1 = 1/2 * ((0 - out_o1)**2)
  error2 = 1/2 * ((1 - out_o2)**2)

  total_error = np.array([error1, error2])

  return np.sum(total_error)


def f(x):
  n_particles = x.shape[0]
  j = [PSO_forward_pass(x[i]) for i in range(n_particles)]
  return np.array(j)

# define the boundaries that the weights can be generated from
dimensions = 8
max_bound = 3 * np.ones(dimensions)
min_bound = - max_bound
bounds = (min_bound, max_bound)

# initialise the swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# create the instance of the PSO optimiser
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)

# run model
cost, pos = optimizer.optimize(f, iters=25)
plot_cost_history(cost_history=optimizer.cost_history)
plt.show()
