#imports
import numpy as np

import sys
print(sys.version)
#My KNN Class
class knn_fatai:
    """
    We use this class to predict the classification of a given data point
    based in a given dataset array. This is my K Nearest Neighbor Algorithm implementation
    """

    def __init__(self, k):
        self.k = k

    
    # A KNN algorithm requires that an entire dataset has to be stored. After this happens,
    # the algorithm will then find the k number of nearest data points to point being tested.
    # a prediction will be based on these neighbors

    def knn_fit(self, data_array, j): #j is the current row index of the classification feature
        for row in data_array:
            row.append(row.pop(j))

        return data_array

    #A method I will be using is the euclidean distance. This will be my distance measurement. 
    #def the euc function
    def euclidean_distance(self, row1, row2):
        #euclidean distance is the square root of the sum of the squares of the differnces
        #instantiate our sum of square of differences
        distance = 0
        # Now I want to iterate by index for each row
        # going to do each direction of each vector separately
        for i in range(0, len(row1)-1):
            distance += (row1[i] - row2[i]**2)
        #now I want to return the square root
        return sqrt(distance)

    def knn_predict(self, data_array, getrow): 
        #data_array is the array produced by knn_fit
        eucdist = []
        #putting the distances in a list 'eucdist'

        #we will put points in a list of arrays 'kneighbors'
        kneighbors = []

        #first I wan to find all the distances
        for row in data_array:
            eucdist.append(self.euclidean_distance(getrow, row))

        #I want to now find the smallest k-distances
        #using numpy.patition is an efficient way to do this
        kdists = []
        kdists = np.partition(eucdist, self.k-1)[:self.k]

        #now I wanna use kdists to compare with eucdist to find the kdists indices
        #will use the index of eucdist to find the appropriate rows from datarows
        for dist in kdists:
            #we'll use the index command to dins the row from datatrows that we want to append
            kneighbors.append(data_arrau[eucdist.index(dist)])

        #now we can use our algorithm to predict a classification
        #our prediction will be the most common classification in our found kneighbors
        #my classification value has already been moved to the end of rach row
        classification= []
        for row in kneighbors:
            classification.append(row[-1])
        return (stats.mode(classification))[0]

    if __name__ == '__main__':
        point = [2.7810836, 2.550537003]
        dataset = [
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]
        ]

model = knn_fatai(3)
model.knn_fit(dataset, 2)

print(point, ' is classified as:')
print(model.knn_predict(dataset, point)) # Should be classified as [0] 
