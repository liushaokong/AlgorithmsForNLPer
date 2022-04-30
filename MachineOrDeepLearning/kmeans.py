"""
this is to show how the kmeans or EM works.
complexity: (kn) * hidden^2
for each one of n vectors, caculate its didtance from the centroids.
"""
import numpy as np
import random
import sys

# simplified version
def KMeans(array, k, max_iter=300):  # k, max_iteration
    """
    array: 1-D array

    steps:
        init clusterAssment, shape=(n, 2), n for len(array), 2 for (Centroids, Distance)
        select k centroids
        init flag clusterChanged=True

        iterate while clusterChanged and within the max_interations.
        while conditions(clusterChanged && <max_iterations):
            for n vectors:
                for k centroids:
                    process

    """
    n = len(array)  # n vectors
    clusterAssment = np.zeros(shape=(n, 2))  # 2-D array for (Centroid, Distance),

    if len(set(array)) < k:  # no enough nums
        return None, None
    
    centroids = random.sample(set(array), k)  # select k numbers
    clusterChanged = True  # to check if any cluster changed
    i = 0  # iterations
    
    while clusterChanged and i < max_iter:  # condition
        clusterChanged = False  # change to False at the beginning, the position is very important
        
        # 2 loops
        for i in range(n):  # iterate over n vectors
            minDist = sys.maxsize  # init dist at the beginning
            minIndex = -1  # init index
            
            for j in range(k):  # iterate over k centers
                distJI = (array[i]-centroids[j]) ** 2  # check dist
                if distJI < minDist:  # find the closest center and index
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # to check if index changed
                clusterChanged = True  # the original index != current index
            
            clusterAssment[i, :] = minIndex, minDist
        
        for cent in range(k):
            # calculate each centroids
            # np.nonzero(clusterAssment[:, 0] == cent) returns clusterAssment meets the condition
            ptsInClust = array[np.nonzero(clusterAssment[:, 0] == cent)[0]]  # E step
            centroids[cent] = np.mean(ptsInClust)  # M step
        
        i += 1
    
    return centroids, clusterAssment


if __name__ == "__main__":
    array = np.random.randint(low=93, high=100, size=10)
    centroids, clusterAssment = KMeans(array, 3)
    print("array: ", array)
    print("centroids: ", centroids)
    print("clusterAssment: \n", clusterAssment)

