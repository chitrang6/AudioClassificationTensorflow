import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from Dataset_Prep import *
from utility import *


XFinalTest = GetFinalTestData()

n_dim = 65
n_classes = 2

n_hidden_units_one = 65
n_hidden_units_two = 65
n_hidden_units_three = 55
learning_rate = 0.01




X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])



Weights_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one]), name = 'w1')
bias_1 = tf.Variable(tf.random_normal([n_hidden_units_one]), name = 'b1')
activation_1 = tf.nn.tanh(tf.matmul(X , Weights_1) + bias_1)

Weights_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two]), name = 'w2')
bias_2 = tf.Variable(tf.random_normal([n_hidden_units_two]), name = 'b2')
#activation_2 = tf.nn.tanh(tf.matmul(activation_1,Weights_2) + bias_2)
activation_2 = tf.nn.sigmoid(tf.matmul(activation_1,Weights_2) + bias_2)



#Weights_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three]))
#bias_3 = tf.Variable(tf.random_normal([n_hidden_units_three]))
#activation_3 = tf.nn.sigmoid(tf.matmul(activation_2,Weights_3) + bias_3)

Weights = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes]), name = 'W')
#Weights = tf.Variable(tf.random_normal([n_hidden_units_three,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]), name = 'B')
yPredbyNN = tf.matmul(activation_2,Weights) + bias
#yPredbyNN = tf.nn.softmax_cross_entropy_with_logits(tf.add(tf.matmul(activation_2, Weights), bias)

new_saver = tf.train.import_meta_graph('mymodelaudio.meta')

with tf.Session() as session:
	new_saver.restore(session, tf.train.latest_checkpoint('./'))
	YPredByNNForUnlabeledData = session.run(tf.argmax(yPredbyNN,1),feed_dict={X: XFinalTest})


print YPredByNNForUnlabeledData

