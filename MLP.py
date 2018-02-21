'''
MLP based on TensorFlow
SkyOrca, UCAS, 2017.10.27
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
class HiddenLayer():
    def __init__(self, input, dim_i_2_h, dim_h_2_next): #input is a numpy matrix[sample, feature]
        W = tf.Variable(tf.random_normal(shape=[dim_i_2_h, dim_h_2_next]))
        b = tf.Variable(tf.random_normal(shape=[dim_h_2_next]))
        init = tf.initialize_all_variables()
        sess.run(init) #?
        output = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        self.input   = input
        self.h_output = output
        self.W       = W
        self.b       = b
        self.params = [self.W, self.b]

class OutputLayer():
    def __init__(self, input, dim_h_2_next, dim_o):  # receive input from the last hidden layer, dim_o may be uesless
        self.W = tf.Variable(tf.random_normal(shape=[dim_h_2_next, dim_o]))
        self.b = tf.Variable(tf.random_normal(shape=[dim_o]))
        init = tf.initialize_all_variables()
        sess.run(init) #?
        output = tf.nn.relu(tf.add(tf.matmul(input, self.W), self.b))
        self.input   = input
        self.output   = output
        self.params  = [self.W, self.b]

class MLP():
    def __init__(self, input_data, input_label, dim_i_2_h, dim_h_2_next, dim_o):
        self.HiddenLayer1 = HiddenLayer(input = input_data, dim_i_2_h=dim_i_2_h, dim_h_2_next=dim_h_2_next)
        self.OutputLayer  = OutputLayer(input = self.HiddenLayer1.h_output, dim_h_2_next=dim_h_2_next, dim_o = dim_o)
        self.loss = (tf.nn.l2_loss(input_label - self.OutputLayer.output))
        self.params = self.HiddenLayer1.params + self.OutputLayer.params

def loadData():
    iris = datasets.load_iris()
    x_vals     = np.array([x[0:3] for x in iris.data])
    #y_vals     = [x[3] for x in iris.data]
    y_vals     = np.array([1. if x==0 else 0. for x in iris.target ])
    #train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    #test_indices  = np.array(list(set(range(len(x_vals)))-set(train_indices)))
    X_train_old = x_vals[:120]
    X_test_old  = x_vals[120:150]
    y_train     = y_vals[:120]
    y_test      = y_vals[120:150]
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train_old)
    X_test  = scaler.transform(X_test_old)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = loadData()
sess = tf.Session()
x_data   = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

train_loss = []
test_loss  = []
batch_size = 10

for i in range(200):
    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = X_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    machine = MLP(x_data, y_target, 3, 5, 1) # build MLP using placeholders
    loss = machine.loss
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    train_loss_tmp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    train_loss.append(train_loss_tmp)
    test_loss_tmp = sess.run(loss, feed_dict={x_data: X_test, y_target: np.transpose([y_test])})
    test_loss.append(test_loss_tmp)
    if (i+1)%20 == 0: print 'Generate %d Loss: %d' % (i+1, train_loss_tmp)
plt.plot(train_loss, 'k--', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.xlabel('Times')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()





'''
1. dim_i_2_h depends on sample size, and dim_h_2_next depends on how many hidden nodes are there in a layer.
dim_o depends on how many nodes in the OutputLayer, for regression usually 1 and for classification? Point out in the
MLP class.
'''
