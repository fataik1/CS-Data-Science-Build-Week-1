import numpy as np
from scipy import stats

class Fatai_KNN:
    """
    This class will be used to predict the classifications of a given data point
    based on a given dataset array. This will be done by comparing pints to its
    K-Nearest-Neighbor in the dataset, where K is provided
    """
    def __init__(self, k):
        self.k = k 

    def euclidean_distance(x1, x2):
        #Euclidean distance is the square root of the sum of squares of the differences
        #look at the formula, calculate the sum of each vector component
        
        #Instantiate our sum of sqaures of differences 
        sumsqdiff = 0
        
        #iterate by index in each row
        for i in range(0, len(x1)-1):
            sumsqdiff += (x1[i] - x2[i])**2
        
        #want to return the square root
        return np.sqrt(sumsqdiff)

    for row in dataset:
        distance.append(euclidean_distance(point, row))
    print(distance)

    def NN_fit(data_array, j): #j is the current index of the classification 
        """
        Method fits the data to the model
        """
        for row in data_array:
            row.append(row.pop(j))
        return data_array

    print(NN_fit(dataset, 0))

    #function we use to get k nearest neighbors
    def get_NN(datarows, getrow, k):
        #put the distances in a list
        eucdist = []

        #put the piunts in a list of arrays 'kneighbors'
        kneighbors = []

        #find all the distances
        for row in datarows:
            eucdist.append(euclidean_distance(getrow, row))

        #find the smallest k-distances
        #use numpy.partition
        kdists = []
        kdists = np.partition(eucdist, k-1)[:k]

        #use kdists to compare eucdist to find kdists indices
        #use the incidices of eucdist to find rows form data rows
        for dist in kdists:
            #use the index command to find the row from datarows that we wanna append
            kneighbors.append(datarows[eucdist.index(dist)])

        return kneighbors

    testneighbors = get_NN(dataset, point, 5)
    print(testneighbors)

    # Our prediction will be the most common classification in our found neighbors
    def knn_predict(neighbors):
        """
        This function will be used to predict classifications
        """
        classification = []
        for row in neighbors:
            classification.append(row[-1])
        return (stats.mode(classification))[0]

    print(knn_predict(testneighbors))

