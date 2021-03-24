#Not a main KNN was working on the side for this. Not finished
import numpy as np
from collections import Counter
 
def euclidean_distance(x1, x2):
    #look at the formula, calculate the sum of each vector component
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        #store our training example
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        #helper method for each of the samples. want to do this for all samples
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        #compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #get the k nearest samples, and want to get the labels
        #sort our distances
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #majority vote, most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]