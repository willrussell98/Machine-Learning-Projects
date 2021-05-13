# code by William Russell

import numpy as np
import matplotlib.pyplot as plt

# Implement a Perceptron Algorithm

class Perceptron():

  # initialise the class and dataset
  def __init__(self, dataset, eta=0.01, n_iter=20):
    self.dataset = dataset
    self.eta = eta
    self.n_iter = n_iter
    self.acFunc = self.activation_function


  # define an activation function
  def activation_function(self, V):
    return np.where(V > 0, 1, -1)

  # predict the activation score based upon the inputs
  def predict(self, X):

    # Vk is the induced local field
    # Sum of the dot products between the weights and the x inputs
    V = np.inner(self.weights, X) + self.bias

    # compute response using the activation function
    # produces the value Yk
    Y = self.acFunc(V)

    return Y

  def plot(self, X):

    # plot the datapoints class 1 and class 2 to show the seperation of the two datasets

    # class 1 has the label -1
    plt.scatter(self.dataset[:,0][0:25], self.dataset[:,1][0:25], color='red', marker='x')

    # class 2 has the label 1
    plt.scatter(self.dataset[:,0][25:49], self.dataset[:,1][25:49], color='blue', marker='o')

    if self.weights[0] == 0 and self.weights[1] == 0 and self.bias == 0:
        print("Hyperplane not properly established")
    
    else:
      # show the range of the x
      x = np.linspace(-100,100)

      # y demonstrates the equation of the hyperplane 
      # self.bias -> Weight 0
      # self.weights[0] -> Weight 1
      # self.weights[1] -> Weight 0
      y = (-(self.bias / self.weights[1]) / (self.bias / self.weights[0])) * x + (-self.bias / self.weights[1])

      # limits of the plot
      plt.xlim([-2, 20])
      plt.ylim([-5, 20])

      # plot the hyperplane
      plt.plot(x, y, color='black', linewidth=2)
    
    plt.show()


  def train(self, X, y):

    # number of features in the training data
    n_features = X.shape[1]

    # initialise the weights and the bias
    self.weights = np.zeros([n_features]) # for all i = 1,2,...,d
    self.bias = 0
    self.errors = []

  
    # iterate through the training procedure 
    for i in range(1, self.n_iter + 1):

      # initialise the error variable to 0
      error = 0
      print(f"Epoch {i}\n")

      # reshuffle the dataset so that the algorithm does not get stuck in cycles
      reshuffle =  np.random.permutation(len(X))
      X = X[reshuffle]
      y = y[reshuffle]

      # index will increment by 1 for each datapoint used (for labelling purposes)
      index = 0

      # where xi = inputs for each features, where d_ = the desired data label
      for xi, desired_label in zip(X, y):
        
        # print the number, data inputs, desired label, predicted label
        print(f"Number: {index} Data Inputs: {xi} Desired Label: {desired_label} Predicted Label: {self.predict(xi)}")

        # print the weights and the bias
        print(f"Weights: {self.weights} Bias: {self.bias}")

        # print the value of the update rule
        update = self.eta * (desired_label - self.predict(xi))
        print(f"Update: {update[0]}")

        # if the update rule is not equal to 0, then there has been a misclassification
        # therefore the weights and bias need to be updated
        if update != 0:

          # increment the errors to show the number of errors accumulated in an epoch
          error += 1

          print("Misclassification\n")
          
          # update the weights and bias
          self.weights += update  *  xi
          self.bias += update

        else:
          print("Correct Classification\n")

        # plot the diagram from the function plot() if the number of features is 2
        if X.shape[1] == 2:
          self.plot(X)
          print("\n\n")

        index += 1

      # append number of errors per epoch to the general self.errors variable
      self.errors.append(error)

      # if statement that will break the loop and stop the algorithm if there
      # are no errors acquired in an epoch, thereby the algorithm has converged
      if error == 0:
        print("The Algorithm has Converged.\n\n")
        break

    # iterate through the error list and print the errors per epoch
    for index, errors in enumerate(self.errors):
      print(f"Epoch {index} \tErrors: {errors}\n")
    
    return 
      
# ignore unnecessary warnings
np.seterr(divide='ignore', invalid='ignore')

# initialise random seed
np.random.seed(14)

# create three experiments:

# X1: 2D Perceptron Algorithm (with 2D plot)

# create two classes with two features and 25 randomly generated samples each, concatenate 
# the two classes to form one dataset which will be represented as X
Class_1 = np.random.multivariate_normal([5,5], [[1,0],[0,1]], 25)
Class_2 = np.random.multivariate_normal([10,10], [[1,0],[0,1]], 25)
X1 = np.concatenate((Class_1, Class_2))

# assign class 1 the label -1 and assign class 2 the label 1 as their y values
label1 = np.full((25, 1), -1, dtype=int)
label2 = np.full((25, 1), 1, dtype=int)
y = np.concatenate((label1, label2))

perceptron_2D = Perceptron(dataset=X1, eta=0.01, n_iter=10)
perceptron_2D.train(X1,y)


# X2: 3D Perceptron Algorithm 

# create two classes with three features and 25 randomly generated samples each, concatenate 
# the two classes to form one dataset which will be represented as X2
Class_3 = np.random.multivariate_normal([5,5,5], [[1,0,0],[0,1,0],[0,0,1]], 25)
Class_4 = np.random.multivariate_normal([10,10,10], [[1,0,0],[0,1,0],[0,0,1]], 25)
X2 = np.concatenate((Class_3, Class_4))

#perceptron_3D = Perceptron(dataset=X2, eta=0.01, n_iter=50)
#perceptron_3D.train(X2,y)

# X3: 4D Perceptron Algorithm 

# create two classes with four features and 25 randomly generated samples each, concatenate 
# the two classes to form one dataset which will be represented as X3
Class_5 = np.random.multivariate_normal([5,5,5,5], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], 25)
Class_6 = np.random.multivariate_normal([10,10,10,10], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], 25)
X3 = np.concatenate((Class_5, Class_6))

#perceptron_4D = Perceptron(dataset=X3, eta=0.01, n_iter=50)
#perceptron_4D.train(X3,y)
