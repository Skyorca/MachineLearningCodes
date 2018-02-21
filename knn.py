#K-NN classification for graphs
# Stanford CS231c

import numpy as np
import operator

def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def knn(input_data, dataset, labels, k):  #note that dataset is the group we generated from the above
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(input_data, (dataset_size, 1)) - dataset
    sqDiff_mat = diff_mat**2
    sqDistances = sqDiff_mat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistance = distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group, labels = create_data_set()
print '[0.5, 0.5]', knn([0.5, 0.5], group, labels, 3)
print '[0.8, 0.8]', knn([0.8, 0.8], group, labels, 3)
print '[1.2, 1.2]', knn([1.2, 1.2], group, labels, 3)



