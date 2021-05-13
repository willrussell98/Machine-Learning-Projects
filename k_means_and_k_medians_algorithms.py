import numpy as np
import matplotlib.pyplot as plt

# define the euclidean distance
def Euclidean(X, Y):

  # Computes the Euclidean distance between X and Y
  return np.linalg.norm(X - Y, axis=1)

# create k-means algorithm
class k_means():

  # initialise the class
  def __init__(self, Dataset, k_clusters=4, n_iter=100):
    X = Dataset
    self.k_clusters = k_clusters
    self.n_iter = n_iter

    # define the shape of the input dataset
    self.n_samples = X.shape[0]
    self.n_features = X.shape[1] 

  # randomly select cluster representatives in the model
  def random_cluster_representative(self, X):
    
    # initialise a number_k_clusters x number_features zeros matrix 
    cluster_rep = np.zeros((self.k_clusters, self.n_features))

    # iterate through each cluster and randomly select a datapoint from all the
    # samples in the dataset that will serve as the random initial representative
    for index in range(self.k_clusters):
      random_rep = X[np.random.choice(range(self.n_samples))]
      cluster_rep[index] = random_rep

    return cluster_rep

  # create clusters
  def create_clusters(self, X, cluster_rep):

    # create an empty array which will store the arrays of each cluster
    clusters = [[] for _ in range(self.k_clusters)]

    # iterate through the dataset X
    for index, Xi in enumerate(X):
      
      # take the euclidean distance of each datapoint from the cluster representatives
      # and append the index of this datapoint to the correct cluster array which has the minimum
      # euclidean distance
      closest_rep = np.argmin(Euclidean(Xi, cluster_rep))
      clusters[closest_rep].append(index)

    return clusters

  # define the new cluster representatives
  def new_cluster_representative(self, X, clusters):

    # create another matrix of zeros: number_k_clusters x number_features 
    cluster_rep = np.zeros((self.k_clusters, self.n_features))

    # iterate through each cluster and take the mean of all the datapoints
    # in the respective cluster, this will then be added to the empty matrix
    # defined above and will represent the new cluster representatives of
    # each new cluster from 0 - k
    for index, cluster in enumerate(clusters):
      new_rep = np.mean(X[cluster], axis=0)
      cluster_rep[index] = new_rep

    return cluster_rep

  # step through each cluster and obtain the label of the cluster it belongs to
  def predict(self, X, clusters):
    cluster_prediction = np.zeros((self.n_samples))

    # iterate through each clusters (0-k)
    for cluster_index, cluster in enumerate(clusters):

      # iterate through each index number in a cluster and update the subsequent index
      # number of the prediction array to the cluster number
      for index_number in cluster:
        cluster_prediction[index_number] = cluster_index

    return cluster_prediction
        
    # evaluation technique to determine the accuracy of the clusters
  def B_Cubed(self, true_labels, cluster_prediction):
  
    # assemble the true_labels and cluster_prediction into two integer arrays
    # that can be compared
    real = np.array(true_labels)
    prediction = cluster_prediction.astype(int)

    # number of labels in the dataset
    number_of_labels = 4

    # initialise confusion matrix full of zeros - number_k_clusters x number_of_labels
    # the rows (k_clusters) represent the prediction of the K_Means Model
    # the columns (number_of_labels) represent the number of actual labels
    confusion_matrix = np.zeros((self.k_clusters, number_of_labels))

    # iterate through the real and predicted values and increment the value of by 1
    # to the given matrix position
    for real_value, predicted_value in zip(real, prediction):
      for predict in range(0, self.k_clusters):
        for actual in range(0, number_of_labels):
          if real_value == actual and predicted_value == predict:
            confusion_matrix[predict][actual] += 1

    # initialise the total_precision, total_recall and total_F_Score sums to 0
    total_precision = 0
    total_recall = 0
    total_F_Score = 0

    # iterate through the real and predicted values to work out the precision, recall and f_score
    # whilst incrementing the total sums of these B_Cubed Measures
    for actual, predict in zip(real, prediction):

      # precision
      precision = (confusion_matrix[predict][actual])/sum(confusion_matrix[predict])
      total_precision += precision

      # recall
      recall = (confusion_matrix[predict][actual])/sum(confusion_matrix[:,actual])
      total_recall += recall

      # f_score
      F_Score = (2 * recall * precision) / (recall + precision)
      total_F_Score += F_Score

    # Find the average precision, recall and f_score
    N = len(real)
    Average_Precision = total_precision / N
    Average_Recall = total_recall / N
    Average_F_Score = total_F_Score / N
    
    return Average_Precision, Average_Recall, Average_F_Score

  # define fit function to train the algorithm to find the optimal new cluster representatives
  def fit(self, X):

    # initialise the random cluster representatives
    cluster_rep = self.random_cluster_representative(X)

    # ground truth clusters - array which holds the true values of each label 
    label1 = [0 for i in range(50)] # animals - 0
    label2 = [1 for i in range(161)] # countries - 1
    label3 = [2 for i in range(58)] # fruits - 2
    label4 = [3 for i in range(60)] # veggies - 3
    true_labels = label1 + label2 + label3 + label4
      
    # iterate through the algorithm n_iter amount of times
    for _ in range(self.n_iter):
      # create new clusters for each iteration
      clusters = self.create_clusters(X, cluster_rep)
    
      # predict the labels of each Xi value in each cluster
      cluster_prediction = self.predict(X, clusters)

      # store the old cluster representative to later calculate if the algorithm converges
      old_cluster_rep = cluster_rep

      # optimise the cluster representatives to find new cluster representatives
      cluster_rep = self.new_cluster_representative(X, clusters)

      # compute the B_Cubed Precision, Recall and F-Score of the cluster predictions
      Average_Precision, Average_Recall, Average_F_Score = self.B_Cubed(true_labels, cluster_prediction)

      # if the sum of the euclidean disatances of the cluster representatives equals 0,
      # then there is no change in the placing of the cluster representatives,
      # therefore the algorithm has converged and we can break from the for loop
      if sum(Euclidean(old_cluster_rep, cluster_rep)) == 0:
        break

    return Average_Precision, Average_Recall, Average_F_Score

# define the manhattan distance
def Manhattan(X, Y):
  # Computes the Manhattan distance between X and Y
  return np.abs(X - Y).sum(axis=1)

# create k_medians algorithm
class k_medians():

  # initialise the class
  def __init__(self, Dataset, k_clusters=4, n_iter=100):
    X = Dataset
    self.k_clusters = k_clusters
    self.n_iter = n_iter

    # define the shape of the input dataset
    self.n_samples = X.shape[0]
    self.n_features = X.shape[1] 

  # randomly select cluster representatives in the model
  def random_cluster_representative(self, X):
    
    # initialise a number_k_clusters x number_features zeros matrix 
    cluster_rep = np.zeros((self.k_clusters, self.n_features))

    # iterate through each cluster and randomly select a datapoint from all the
    # samples in the dataset that will serve as the random initial representative
    for index in range(self.k_clusters):
      random_rep = X[np.random.choice(range(self.n_samples))]
      cluster_rep[index] = random_rep

    return cluster_rep

  # create clusters
  def create_clusters(self, X, cluster_rep):

    # create an empty array which will store the arrays of each cluster
    clusters = [[] for _ in range(self.k_clusters)]

    # iterate through the dataset X
    for index, Xi in enumerate(X):
      # take the manhattan distance of each datapoint from the cluster representatives
      # and append the index of this datapoint to the correct cluster array which has the minimum
      # manhattan distance
      closest_rep = np.argmin(Manhattan(Xi, cluster_rep))
      clusters[closest_rep].append(index)

    return clusters

  # define the new cluster representatives
  def new_cluster_representative(self, X, clusters):

    # create another matrix of zeros: number_k_clusters x number_features 
    cluster_rep = np.zeros((self.k_clusters, self.n_features))

    # iterate through each cluster and take the median of all the datapoints
    # in the respective cluster, this will then be added to the empty matrix
    # defined above and will represent the new cluster representatives of
    # each new cluster from 0 - k
    for index, cluster in enumerate(clusters):
      new_rep = np.median(X[cluster], axis=0)
      cluster_rep[index] = new_rep

    return cluster_rep

  # step through each cluster and obtain the label of the cluster it belongs to
  def predict(self, X, clusters):
    cluster_prediction = np.zeros((self.n_samples))

    # iterate through each clusters (0-k)
    for cluster_index, cluster in enumerate(clusters):

      # iterate through each index number in a cluster and update the subsequent index
      # number of the prediction array to the cluster number
      for index_number in cluster:
        cluster_prediction[index_number] = cluster_index

    return cluster_prediction
        
    # evaluation technique to determine the accuracy of the clusters
  def B_Cubed(self, true_labels, cluster_prediction):
  
    # assemble the true_labels and cluster_prediction into two integer arrays
    # that can be compared
    real = np.array(true_labels)
    prediction = cluster_prediction.astype(int)

    # number of labels in the dataset
    number_of_labels = 4

    # initialise confusion matrix full of zeros - number_k_clusters x number_of_labels
    # the rows (k_clusters) represent the prediction of the K_Means Model
    # the columns (number_of_labels) represent the number of actual labels
    confusion_matrix = np.zeros((self.k_clusters, number_of_labels))

    # iterate through the real and predicted values and increment the value of by 1
    # to the given matrix position
    for real_value, predicted_value in zip(real, prediction):
      for predict in range(0, self.k_clusters):
        for actual in range(0, number_of_labels):
          if real_value == actual and predicted_value == predict:
            confusion_matrix[predict][actual] += 1

    # initialise the total_precision, total_recall and total_F_Score sums to 0
    total_precision = 0
    total_recall = 0
    total_F_Score = 0

    # iterate through the real and predicted values to work out the precision, recall and f_score
    # whilst incrementing the total sums of these B_Cubed Measures
    for actual, predict in zip(real, prediction):

      # precision
      precision = (confusion_matrix[predict][actual])/sum(confusion_matrix[predict])
      total_precision += precision

      # recall
      recall = (confusion_matrix[predict][actual])/sum(confusion_matrix[:,actual])
      total_recall += recall

      # f_score
      F_Score = (2 * recall * precision) / (recall + precision)
      total_F_Score += F_Score

    # Find the average precision, recall and f_score
    N = len(real)
    Average_Precision = total_precision / N
    Average_Recall = total_recall / N
    Average_F_Score = total_F_Score / N
    
    return Average_Precision, Average_Recall, Average_F_Score

  # define fit function to train the algorithm to find the optimal new cluster representatives
  def fit(self, X):

    # initialise the random cluster representatives
    cluster_rep = self.random_cluster_representative(X)

    # ground truth clusters - array which holds the true values of each label 
    label1 = [0 for i in range(50)] # animals - 0
    label2 = [1 for i in range(161)] # countries - 1
    label3 = [2 for i in range(58)] # fruits - 2
    label4 = [3 for i in range(60)] # veggies - 3
    true_labels = label1 + label2 + label3 + label4
      
    # iterate through the algorithm n_iter amount of times
    for _ in range(self.n_iter):
      # create new clusters for each iteration
      clusters = self.create_clusters(X, cluster_rep)
    
      # predict the labels of each Xi value in each cluster
      cluster_prediction = self.predict(X, clusters)

      # store the old cluster representative to later calculate if the algorithm converges
      old_cluster_rep = cluster_rep

      # optimise the cluster representatives to find new cluster representatives
      cluster_rep = self.new_cluster_representative(X, clusters)

      # compute the B_Cubed Precision, Recall and F-Score of the cluster predictions
      Average_Precision, Average_Recall, Average_F_Score = self.B_Cubed(true_labels, cluster_prediction)

      # if the sum of the euclidean disatances of the cluster representatives equals 0,
      # then there is no change in the placing of the cluster representatives,
      # therefore the algorithm has converged and we can break from the for loop
      if sum(Euclidean(old_cluster_rep, cluster_rep)) == 0:
        break

    return Average_Precision, Average_Recall, Average_F_Score

# download the data from the files
def load_data(fname):
    features = []
    
    with open(fname) as F:
        for line in F:
            p = line.strip().split()
            features.append(np.array(p[1:], float))
            
    return np.array(features)

animal_dataset = load_data("animals")
countries_dataset = load_data("countries")
fruits_dataset = load_data("fruits")
veggies_dataset = load_data("veggies")

# create dataset
dataset = np.concatenate((animal_dataset, countries_dataset, fruits_dataset, veggies_dataset))

# normalise the data into unit L2 length
def L2_Normalisation(data):
  # calculate the L2 vector norm
  normalisation = np.linalg.norm(data, ord=2, axis=0)
  # defined each datapoint in the data by the L2 vector norm
  normalised_data = data/normalisation
  return normalised_data

# create the L2 normalised dataset
L2_normalised_data = L2_Normalisation(dataset)

import warnings
warnings.simplefilter("ignore")

# plot the B-Cubed evaluative measures of the K-Means algorithm on one single bar chart
def K_Means_Plot_Accuracies():

  Precision = []
  Recall = []
  F_Score = []

  # K_Means
  # K = 1
  K_Means1 = k_means(Dataset=dataset, k_clusters=1, n_iter=100)
  Precision1, Recall1, F_Score1 = K_Means1.fit(dataset)

  Precision.append(Precision1)
  Recall.append(Recall1)
  F_Score.append(F_Score1)

  # K = 2
  K_Means2 = k_means(Dataset=dataset, k_clusters=2, n_iter=100)
  Precision2, Recall2, F_Score2 = K_Means2.fit(dataset)

  Precision.append(Precision2)
  Recall.append(Recall2)
  F_Score.append(F_Score2)

  # K = 3
  K_Means3 = k_means(Dataset=dataset, k_clusters=3, n_iter=100)
  Precision3, Recall3, F_Score3 = K_Means3.fit(dataset)
  Precision.append(Precision3)
  Recall.append(Recall3)
  F_Score.append(F_Score3)

  # K = 4
  K_Means4 = k_means(Dataset=dataset, k_clusters=4, n_iter=100)
  Precision4, Recall4, F_Score4 = K_Means4.fit(dataset)

  Precision.append(Precision4)
  Recall.append(Recall4)
  F_Score.append(F_Score4)

  # K = 5
  K_Means5 = k_means(Dataset=dataset, k_clusters=5, n_iter=100)
  Precision5, Recall5, F_Score5 = K_Means5.fit(dataset)

  Precision.append(Precision5)
  Recall.append(Recall5)
  F_Score.append(F_Score5)

  # K = 6
  K_Means6 = k_means(Dataset=dataset, k_clusters=6, n_iter=100)
  Precision6, Recall6, F_Score6 = K_Means6.fit(dataset)

  Precision.append(Precision6)
  Recall.append(Recall6)
  F_Score.append(F_Score6)

  # K = 7
  K_Means7 = k_means(Dataset=dataset, k_clusters=7, n_iter=100)
  Precision7, Recall7, F_Score7 = K_Means7.fit(dataset)

  Precision.append(Precision7)
  Recall.append(Recall7)
  F_Score.append(F_Score7)

  # K = 8
  K_Means8 = k_means(Dataset=dataset, k_clusters=8, n_iter=100)
  Precision8, Recall8, F_Score8 = K_Means8.fit(dataset)

  Precision.append(Precision8)
  Recall.append(Recall8)
  F_Score.append(F_Score8)

  # K = 9
  K_Means9 = k_means(Dataset=dataset, k_clusters=9, n_iter=100)
  Precision9, Recall9, F_Score9 = K_Means9.fit(dataset)

  Precision.append(Precision9)
  Recall.append(Recall9)
  F_Score.append(F_Score9)

  # create bar chart
  labels = ['K = 1', 'K = 2', 'K = 3', 'K = 4', 'K = 5', 'K = 6', 'K = 7', 'K = 8', 'K = 9']
  
  # set width of bars
  barWidth = 0.2
 
  # set position of bar on X axis
  position1 = np.arange(len(labels))
  position2 = [x + barWidth for x in position1]
  position3 = [x + barWidth for x in position2]

  fig, ax = plt.subplots(figsize=(12,8))

  Precision_Chart = ax.bar(position1, Precision, barWidth, color='red', label='Precision')
  Recall_Chart = ax.bar(position2, Recall, barWidth, color='blue', label='Recall')
  F_Score_Chart = ax.bar(position3, F_Score, barWidth, color='green', label='F_Score')

  ax.set_xlabel('Number of K Clusters', fontsize=15, labelpad=10.0)
  ax.set_ylabel('B-Cubed Scores', fontsize=15, labelpad=10.0)
  ax.set_title('K-Means Algorithm', fontsize=15)
  ax.set_xticks(position2)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

  # show results of algorithm
  print("\n")
  print(f"Precision: {Precision}")
  print(f"Recall: {Recall}")
  print(f"F_Score: {F_Score}")



# plot the B-Cubed evaluative measures for the L2 Normalised K-Means
# algorithm on one single bar chart
def L2_Normalised_K_Means_Plot_Accuracies():

  Precision = []
  Recall = []
  F_Score = []

  # K_Means
  # K = 1
  K_Means1 = k_means(Dataset=L2_normalised_data, k_clusters=1, n_iter=100)
  Precision1, Recall1, F_Score1 = K_Means1.fit(L2_normalised_data)

  Precision.append(Precision1)
  Recall.append(Recall1)
  F_Score.append(F_Score1)

  # K = 2
  K_Means2 = k_means(Dataset=L2_normalised_data, k_clusters=2, n_iter=100)
  Precision2, Recall2, F_Score2 = K_Means2.fit(L2_normalised_data)

  Precision.append(Precision2)
  Recall.append(Recall2)
  F_Score.append(F_Score2)

  # K = 3
  K_Means3 = k_means(Dataset=L2_normalised_data, k_clusters=3, n_iter=100)
  Precision3, Recall3, F_Score3 = K_Means3.fit(L2_normalised_data)
  Precision.append(Precision3)
  Recall.append(Recall3)
  F_Score.append(F_Score3)

  # K = 4
  K_Means4 = k_means(Dataset=L2_normalised_data, k_clusters=4, n_iter=100)
  Precision4, Recall4, F_Score4 = K_Means4.fit(L2_normalised_data)

  Precision.append(Precision4)
  Recall.append(Recall4)
  F_Score.append(F_Score4)

  # K = 5
  K_Means5 = k_means(Dataset=L2_normalised_data, k_clusters=5, n_iter=100)
  Precision5, Recall5, F_Score5 = K_Means5.fit(L2_normalised_data)

  Precision.append(Precision5)
  Recall.append(Recall5)
  F_Score.append(F_Score5)

  # K = 6
  K_Means6 = k_means(Dataset=L2_normalised_data, k_clusters=6, n_iter=100)
  Precision6, Recall6, F_Score6 = K_Means6.fit(L2_normalised_data)

  Precision.append(Precision6)
  Recall.append(Recall6)
  F_Score.append(F_Score6)

  # K = 7
  K_Means7 = k_means(Dataset=L2_normalised_data, k_clusters=7, n_iter=100)
  Precision7, Recall7, F_Score7 = K_Means7.fit(L2_normalised_data)

  Precision.append(Precision7)
  Recall.append(Recall7)
  F_Score.append(F_Score7)

  # K = 8
  K_Means8 = k_means(Dataset=L2_normalised_data, k_clusters=8, n_iter=100)
  Precision8, Recall8, F_Score8 = K_Means8.fit(L2_normalised_data)

  Precision.append(Precision8)
  Recall.append(Recall8)
  F_Score.append(F_Score8)

  # K = 9
  K_Means9 = k_means(Dataset=L2_normalised_data, k_clusters=9, n_iter=100)
  Precision9, Recall9, F_Score9 = K_Means9.fit(L2_normalised_data)

  Precision.append(Precision9)
  Recall.append(Recall9)
  F_Score.append(F_Score9)

  # create bar chart
  labels = ['K = 1', 'K = 2', 'K = 3', 'K = 4', 'K = 5', 'K = 6', 'K = 7', 'K = 8', 'K = 9']
  
  # set width of bars
  barWidth = 0.2
 
  # set position of bar on X axis
  position1 = np.arange(len(labels))
  position2 = [x + barWidth for x in position1]
  position3 = [x + barWidth for x in position2]

  fig, ax = plt.subplots(figsize=(12,8))

  Precision_Chart = ax.bar(position1, Precision, barWidth, color='red', label='Precision')
  Recall_Chart = ax.bar(position2, Recall, barWidth, color='blue', label='Recall')
  F_Score_Chart = ax.bar(position3, F_Score, barWidth, color='green', label='F_Score')

  ax.set_xlabel('Number of K Clusters', fontsize=15, labelpad=10.0)
  ax.set_ylabel('B-Cubed Scores', fontsize=15, labelpad=10.0)
  ax.set_title('L2 Normalised K-Means Algorithm', fontsize=15)
  ax.set_xticks(position2)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

  # show results of algorithm
  print("\n")
  print(f"Precision: {Precision}")
  print(f"Recall: {Recall}")
  print(f"F_Score: {F_Score}")



# plot the B-Cubed evaluative measures for the K-Medians algorithm on one single bar chart
def K_Medians_Plot_Accuracies():

  Precision = []
  Recall = []
  F_Score = []

  # K_Means
  # K = 1
  K_Medians = k_medians(Dataset=dataset, k_clusters=1, n_iter=100)
  Precision1, Recall1, F_Score1 = K_Medians.fit(dataset)

  Precision.append(Precision1)
  Recall.append(Recall1)
  F_Score.append(F_Score1)

  # K = 2
  K_Medians2 = k_medians(Dataset=dataset, k_clusters=2, n_iter=100)
  Precision2, Recall2, F_Score2 = K_Medians2.fit(dataset)
  
  Precision.append(Precision2)
  Recall.append(Recall2)
  F_Score.append(F_Score2)

  # K = 3
  K_Medians3 = k_medians(Dataset=dataset, k_clusters=3, n_iter=100)
  Precision3, Recall3, F_Score3 = K_Medians3.fit(dataset)

  
  Precision.append(Precision3)
  Recall.append(Recall3)
  F_Score.append(F_Score3)

  # K = 4
  K_Medians4 = k_medians(Dataset=dataset, k_clusters=4, n_iter=100)
  Precision4, Recall4, F_Score4 = K_Medians4.fit(dataset)

  Precision.append(Precision4)
  Recall.append(Recall4)
  F_Score.append(F_Score4)

  # K = 5
  K_Medians5 = k_medians(Dataset=dataset, k_clusters=5, n_iter=100)
  Precision5, Recall5, F_Score5 = K_Medians5.fit(dataset)

  Precision.append(Precision5)
  Recall.append(Recall5)
  F_Score.append(F_Score5)

  # K = 6
  K_Medians6 = k_medians(Dataset=dataset, k_clusters=6, n_iter=100)
  Precision6, Recall6, F_Score6 = K_Medians6.fit(dataset)

  Precision.append(Precision6)
  Recall.append(Recall6)
  F_Score.append(F_Score6)

  # K = 7
  K_Medians7 = k_medians(Dataset=dataset, k_clusters=7, n_iter=100)
  Precision7, Recall7, F_Score7 = K_Medians7.fit(dataset)

  Precision.append(Precision7)
  Recall.append(Recall7)
  F_Score.append(F_Score7)

  # K = 8
  K_Medians8 = k_medians(Dataset=dataset, k_clusters=8, n_iter=100)
  Precision8, Recall8, F_Score8 = K_Medians8.fit(dataset)

  Precision.append(Precision8)
  Recall.append(Recall8)
  F_Score.append(F_Score8)

  # K = 9
  K_Medians9 = k_medians(Dataset=dataset, k_clusters=9, n_iter=100)
  Precision9, Recall9, F_Score9 = K_Medians9.fit(dataset)

  Precision.append(Precision9)
  Recall.append(Recall9)
  F_Score.append(F_Score9)

  # create bar chart
  labels = ['K = 1', 'K = 2', 'K = 3', 'K = 4', 'K = 5', 'K = 6', 'K = 7', 'K = 8', 'K = 9']
  
  # set width of bars
  barWidth = 0.2
 
  # set position of bar on X axis
  position1 = np.arange(len(labels))
  position2 = [x + barWidth for x in position1]
  position3 = [x + barWidth for x in position2]

  fig, ax = plt.subplots(figsize=(12,8))

  Precision_Chart = ax.bar(position1, Precision, barWidth, color='red', label='Precision')
  Recall_Chart = ax.bar(position2, Recall, barWidth, color='blue', label='Recall')
  F_Score_Chart = ax.bar(position3, F_Score, barWidth, color='green', label='F_Score')

  ax.set_xlabel('Number of K Clusters', fontsize=15, labelpad=10.0)
  ax.set_ylabel('B-Cubed Scores', fontsize=15, labelpad=10.0)
  ax.set_title('K-Medians Algorithm', fontsize=15)
  ax.set_xticks(position2)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

  # show results of algorithm
  print("\n")
  print(f"Precision: {Precision}")
  print(f"Recall: {Recall}")
  print(f"F_Score: {F_Score}")


# plot the B-Cubed evaluative measures for the L2 Normalised K-Medians algorithm 
# on one single bar chart
def L2_Normalised_K_Medians_Plot_Accuracies():

  Precision = []
  Recall = []
  F_Score = []

  # K_Means
  # K = 1
  K_Medians = k_medians(Dataset=L2_normalised_data, k_clusters=1, n_iter=100)
  Precision1, Recall1, F_Score1 = K_Medians.fit(L2_normalised_data)

  Precision.append(Precision1)
  Recall.append(Recall1)
  F_Score.append(F_Score1)

  # K = 2
  K_Medians2 = k_medians(Dataset=L2_normalised_data, k_clusters=2, n_iter=100)
  Precision2, Recall2, F_Score2 = K_Medians2.fit(L2_normalised_data)
  
  Precision.append(Precision2)
  Recall.append(Recall2)
  F_Score.append(F_Score2)

  # K = 3
  K_Medians3 = k_medians(Dataset=L2_normalised_data, k_clusters=3, n_iter=100)
  Precision3, Recall3, F_Score3 = K_Medians3.fit(L2_normalised_data)

  Precision.append(Precision3)
  Recall.append(Recall3)
  F_Score.append(F_Score3)

  # K = 4
  K_Medians4 = k_medians(Dataset=L2_normalised_data, k_clusters=4, n_iter=100)
  Precision4, Recall4, F_Score4 = K_Medians4.fit(L2_normalised_data)

  Precision.append(Precision4)
  Recall.append(Recall4)
  F_Score.append(F_Score4)

  # K = 5
  K_Medians5 = k_medians(Dataset=L2_normalised_data, k_clusters=5, n_iter=100)
  Precision5, Recall5, F_Score5 = K_Medians5.fit(L2_normalised_data)

  Precision.append(Precision5)
  Recall.append(Recall5)
  F_Score.append(F_Score5)

  # K = 6
  K_Medians6 = k_medians(Dataset=L2_normalised_data, k_clusters=6, n_iter=100)
  Precision6, Recall6, F_Score6 = K_Medians6.fit(L2_normalised_data)

  Precision.append(Precision6)
  Recall.append(Recall6)
  F_Score.append(F_Score6)

  # K = 7
  K_Medians7 = k_medians(Dataset=L2_normalised_data, k_clusters=7, n_iter=100)
  Precision7, Recall7, F_Score7 = K_Medians7.fit(L2_normalised_data)

  Precision.append(Precision7)
  Recall.append(Recall7)
  F_Score.append(F_Score7)

  # K = 8
  K_Medians8 = k_medians(Dataset=L2_normalised_data, k_clusters=8, n_iter=100)
  Precision8, Recall8, F_Score8 = K_Medians8.fit(L2_normalised_data)

  Precision.append(Precision8)
  Recall.append(Recall8)
  F_Score.append(F_Score8)

  # K = 9
  K_Medians9 = k_medians(Dataset=L2_normalised_data, k_clusters=9, n_iter=100)
  Precision9, Recall9, F_Score9 = K_Medians9.fit(L2_normalised_data)

  Precision.append(Precision9)
  Recall.append(Recall9)
  F_Score.append(F_Score9)

  # create bar chart
  labels = ['K = 1', 'K = 2', 'K = 3', 'K = 4', 'K = 5', 'K = 6', 'K = 7', 'K = 8', 'K = 9']
  
  # set width of bars
  barWidth = 0.2
 
  # set position of bar on X axis
  position1 = np.arange(len(labels))
  position2 = [x + barWidth for x in position1]
  position3 = [x + barWidth for x in position2]

  fig, ax = plt.subplots(figsize=(12,8))

  Precision_Chart = ax.bar(position1, Precision, barWidth, color='red', label='Precision')
  Recall_Chart = ax.bar(position2, Recall, barWidth, color='blue', label='Recall')
  F_Score_Chart = ax.bar(position3, F_Score, barWidth, color='green', label='F_Score')

  ax.set_xlabel('Number of K Clusters', fontsize=15, labelpad=10.0)
  ax.set_ylabel('B-Cubed Scores', fontsize=15, labelpad=10.0)
  ax.set_title('L2 Normalised K-Medians Algorithm', fontsize=15)
  ax.set_xticks(position2)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

  # show results of algorithm
  print("\n")
  print(f"Precision: {Precision}")
  print(f"Recall: {Recall}")
  print(f"F_Score: {F_Score}")


if __name__ == "__main__":
  K_Means_Plot_Accuracies()
  print("\n\n\n")
  L2_Normalised_K_Means_Plot_Accuracies()
  print("\n\n\n")
  K_Medians_Plot_Accuracies()
  print("\n\n\n")
  L2_Normalised_K_Medians_Plot_Accuracies()
