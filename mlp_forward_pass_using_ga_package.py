# install geneticalgorithm on device

import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga

# Multi-Layer Perceptron using backpropagation
X_ga = np.array([[0.05258131, 0.44914306]])

# Multi-Layer Perceptron using Genetic Algorithms
def f(W):

  X = np.array([[0.05258131, 0.44914306]])

  # first layer
  net_h1 = (X[0][0] * W[0]) + (X[0][1] * W[1]) + 0.35
  out_h1 = 1/(1 + np.exp(-net_h1))

  net_h2 = (X[0][0] * W[2]) + (X[0][1] * W[3]) + 0.35
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

# define the boundaries that the weights can be generated from
varbound=np.array([[-3,3]]*8)

# set the algorithm parameters
algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.02,\
                   'crossover_probability': 1,\
                   'parents_portion': 0.02,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':1}

# define the model of the genetic algorithm
model=ga(function=f,\
            dimension=8,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

model.run()
