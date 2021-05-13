import numpy as np
import matplotlib.pyplot as plt

# create a multi-layer perceptron algorithm

class MultiLayerPerceptron():

  # initialise the class
  def __init__(self, dataset, n_inputs=2, n_hidden_neurons=[2], n_outputs=2, eta=0.01, n_iter=10, sigmoid=True):
    X = dataset
    self.n_inputs = n_inputs
    self.n_hidden_neurons = n_hidden_neurons
    self.n_outputs = n_outputs
    self.eta = eta
    self.n_iter = n_iter
    self.sigmoid = sigmoid
    self.acFunc1 = self.sigmoid_function
    self.acFunc2 = self.tanh_function
    self.n_features = X.shape[1]

  # define the sigmoid activation function 
  def sigmoid_function(self, V):
    return 1/(1 + np.exp(-V))

  # define the hyperbolic tangent activation function 
  def tanh_function(self, V):
    return 2/(1 + np.exp(-2 * V)) -1

  # define the forward pass function
  def predict(self, X, weights, bias):

    # will pass the values V through the tan function if sigmoid is false
    if self.sigmoid == False:
      V = np.inner(weights, X) + bias
      Y = self.acFunc2(V)

    # will pass the values V through the sigmoid function if sigmoid is true
    elif self.sigmoid == True:
      V = np.inner(weights, X) + bias
      Y = self.acFunc1(V)

    return Y

  # define the neuron error function
  def neuron_error(self, desired_target, actual_output):
    errors = 1/2 * ((desired_target - actual_output) ** 2)
    return errors

  # define the network error function
  def network_error(self, errors):
    network_errors = errors.sum()
    return network_errors

  # define delta function for backpropagation of the output layer
  def delta_output(self, target, output):
    delta = (-(target - output)) * (output * (1 - output))
    return delta

  # define the fit function
  def forward_pass(self, X, weights):

    # first weights = (number_features x number_hidden_neurons)
    W1 = np.array([[weights[0:4]]]).reshape(self.n_features, self.n_hidden_neurons[0]) # for all i = 1,2,...,d

    # first fixed bias
    B1 = 0.35

    # output of hidden neurons
    H = self.predict(X, W1, B1).reshape(1,self.n_hidden_neurons[0])

    # second weights = (number_hidden_neurons x number_outputs)
    W2 = np.array([[weights[4:8]]]).reshape(self.n_hidden_neurons[0], self.n_outputs)

    # second fixed bias
    B2 = 0.6

    # output values of neurons in output layer
    O = self.predict(H, W2, B2).reshape(1, self.n_outputs)
    
    # desired outputs
    D = np.array([[0],[1]]).reshape(1, self.n_outputs)

    # individual errors of the neurons
    neuron_error = self.neuron_error(D,O)

    # total error of the network
    total_error = self.network_error(neuron_error)


    return D, O, H, W1, W2, total_error, neuron_error


  # define the backward pass function for backpropagation
  def backward_pass(self, X, D, O, H, W1, W2, total_error):

    ################# Output Layer #############################

    New_Weights_Output = []

    # iterate through the number of outputs to get the corresnponding delta
    # values for each output neuron
    for output_neuron in range(self.n_outputs):
      delta = self.delta_output(D[0][output_neuron], O[0][output_neuron])

      # iterate through the old weights to update the respective weight
      # using the equation W = W - eta * (dTotalError/dRespectiveWeight)  

      for index, weight in enumerate(W2[output_neuron]):
        update = self.eta * (delta * H[0][index])
        weight = weight - update
        New_Weights_Output.append(weight)


     ################# Hidden Layer #############################

    New_Weights_Hidden = []

    for hidden_neuron in range(self.n_hidden_neurons[0]):

      total_errors = 0

      # calculate the delta values from both output neurons and multiply it
      # by the previous values of the weights that connect the output neurons
      # to the hidden neurons (Ex W2)
      for output_neuron in range(self.n_outputs):

          delta = self.delta_output(D[0][output_neuron], O[0][output_neuron])
          error = delta * W2[output_neuron][hidden_neuron]
          total_errors += error
  
      # iterate through the weights that connect the input layers to the hidden layers and
      # update the weights using the equation W = W - eta * (dTotalError/dRespectiveWeight)  
      for index, weight in enumerate(W1[hidden_neuron]):

        # output of hidden layer
        output_hidden_layer = H[0][hidden_neuron] * (1 - H[0][hidden_neuron])

        # input values
        net_Input = X[index]

        update = self.eta * (total_errors * output_hidden_layer * net_Input)
        weight = weight - update
        New_Weights_Hidden.append(weight)

    # add both weight array together to get [W1,W2,...,W8]
    New_Weights = New_Weights_Hidden + New_Weights_Output

    return New_Weights

  # plot the network error of the algorithm
  def plot(self, X, network_errors):
    plt.plot(range(0,(len(X)*self.n_iter)), network_errors, color='red')
    plt.xlabel('Number of time steps')
    plt.ylabel('Training Network Error')
    plt.show()


  # define a fit function that will train the data
  def train(self, X):

    # initialise random weights, the number of weights:
    # Input Layer -> Hidden Layer: (number_inputs * number_hidden_neurons) weights
    # Hidden Layer -> Output Layer: (number_hidden_neurons * number_outputs) weights
    weights = np.random.normal(loc=0.2, scale=0.05, size=(1, (self.n_inputs * self.n_hidden_neurons[0])+(self.n_hidden_neurons[0]*self.n_outputs)))[0]

    network_error = []

    for epoch in range(self.n_iter):
      #print(f"Epoch {epoch + 1}\n\n\n")

      # optimise the weights through backpropagation by iterating through the dataset
      for training_network in X:

        forward_propagate = self.forward_pass(training_network, weights)
        #print(f"first weights:\n {forward_propagate[3]}\n")  # first weights [[W1,W2],[W3,W4]]
        #print(f"second weights:\n {forward_propagate[4]}\n") # second weights [[W5,W6],[W7,W8]]
        #print(f"Desired Neuron Outputs:\n {forward_propagate[0]}\n") # desired target [O1 Target, O2 Target]
        #print(f"Hidden Neuron Outputs: {forward_propagate[2]}\n") # hidden layer values [H1 Output, H2 Output]
        #print(f"Actual Neuron Outputs:\n {forward_propagate[1]}\n") # output layer values [O1 Output, O2 Output]
        #print(f"Neural Error: {forward_propagate[6]}\n") # total error of the network
        #print(f"Total Network Error: {round(forward_propagate[5], 10)}\n") # total error of the network
        network_error.append(round(forward_propagate[5], 10))
        back_propagate = self.backward_pass(training_network, forward_propagate[0], forward_propagate[1], forward_propagate[2], forward_propagate[3], forward_propagate[4], forward_propagate[5])

        # print out the new weights of the training_network
        #for index, weight in enumerate(back_propagate):
        #  print(f"Weight {index + 1}: {weight}")

        weights = back_propagate
        #print("\n\n\n")

    # plot the graph
    self.plot(X, network_error)


# generate random dataset
np.random.seed(56)
X1 = np.random.multivariate_normal([0.05,0.3], [[0.01,0],[0,0.05]], 10)
X2 = np.random.multivariate_normal([0.4,0.4], [[0.1,0],[0,0.1]], 10)
X3 = np.random.multivariate_normal([0.15,0.15], [[0.1,0],[0,0.25]], 10)
X4 = np.random.multivariate_normal([0.2,0.15], [[0.03,0],[0,0.05]], 10)
X = np.concatenate((X1, X2, X3, X4))

MLP1 = MultiLayerPerceptron(dataset=X, n_inputs=2, n_hidden_neurons=[2], n_outputs=2, eta=0.01, n_iter=50, sigmoid=True)
MLP1.train(X)
print("\n\n\n")
MLP2 = MultiLayerPerceptron(dataset=X, n_inputs=2, n_hidden_neurons=[2], n_outputs=2, eta=0.01, n_iter=50, sigmoid=False)
MLP2.train(X)
