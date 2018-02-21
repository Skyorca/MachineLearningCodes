import numpy as np
import time
print '\nMLAlgo1: Different Versions of Gradient Descent by ORCA\n'
# import data

def LoadDataSet():
    fr = open('testset1.txt')
    data = []      #like placeholder in TF
    label = []
    for line in fr.readlines():
        line_row = line.strip().split()  #split testset as different rows
        data.append([1.0, float(line_row[0]), float(line_row[1])])     #1.0 for easy doing
        label.append(int(line_row[2]))  #why ([int(line_row[2])]) is wrong when transposing?
    return data, label

#so we get two list: data(100*1)(every elem is [x,y,z]), label(100*1)
#but in python grammer there is no matrix, so we need to turn them into matrix by Numpy

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def BatchGradientDescent(data_in, label_in):   #remeber: here is not real matrix
    data_matrix = np.mat(data_in)
    label_matrix = np.mat(label_in).transpose()

    m, n = np.shape(data_matrix)
    learning_rate = 0.05
    training_time = 1000
    weight = np.ones([n,1])  #weight is a 3*1 matrix
    print m,n
    for i in range(training_time):
        h = sigmoid(data_matrix*weight)  #h is a 100*1 matrix
        error = (label_matrix-h)  #error is a 100*1 matrix
        weight = weight+learning_rate*data_matrix.transpose()*error  #update weight
    return weight
start = time.clock()
print 'ver1: Batch Gradient Descent\n'
data_in, label_in = LoadDataSet()
print 'weight=:', BatchGradientDescent(data_in, label_in), 'using time', time.clock()-start, 'seconds'


def StomaticGradientDescent(data_in, label_in):
    m, n = np.shape(data_in)
    learning_rate = 0.05
    weight = np.ones(n)  #weight is a 3*1 matrix
    for j in range(m):
        h = sigmoid(np.sum(data_in[j]*weight))  #h is a 100*1 matrix
        error = (label_in[j]-h)  #error is a 100*1 matrix
        weight = weight+learning_rate*error*data_in[j]
    return weight
start = time.clock()
print 'ver2: Stomatic Gradient Descent\n'
data_in, label_in = LoadDataSet()
print 'weight=:', StomaticGradientDescent(np.array(data_in), label_in)
print 'using time', time.clock()-start, 'seconds'

def AdvancedStomaticGradientDescent(data_in, label_in, iter_time):
    m, n = np.shape(data_in)
    weight = np.ones(n)
    for i in range(iter_time):
        dataset = range(m)
        for j in range(m):
            learning_rate = 4/(1.+i+j)+0.05
            rand_index = int(np.random.uniform(0, len(dataset), 1))
            h = sigmoid(np.sum(data_in[rand_index]*weight))
            error = label_in[rand_index]-h
            wieght = weight+learning_rate*error*data_in[rand_index]
            del(dataset[rand_index])
    return weight
start = time.clock()
print 'ver3: Adanced Stomatic Gradient Descent\n'
print '1. changeable learning rate  2. randomly pick up data in one iteration\n'
data_in, label_in = LoadDataSet()
print 'weight=:', AdvancedStomaticGradientDescent(np.array(data_in), label_in, 500)
print 'using time', time.clock()-start, 'seconds'
