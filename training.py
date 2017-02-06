import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from Dataset_Prep import *
from utility import *


XFinalTest = GetFinalTestData()

X_train, X_test, y_train, y_test = GetDatasetSplit()



training_epochs = 2000
n_dim = X_train.shape[1]
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

tf.add_to_collection('vars', Weights_1)
tf.add_to_collection('vars', bias_1)

tf.add_to_collection('vars', Weights_2)
tf.add_to_collection('vars', bias_2)

tf.add_to_collection('vars', Weights)
tf.add_to_collection('vars', bias)


saver = tf.train.Saver()


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yPredbyNN, Y))
#cost_function = -tf.reduce_sum(Y * tf.log(yPredbyNN))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.75).minimize(loss)

CorrectPred = tf.equal(tf.argmax(yPredbyNN,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(CorrectPred, tf.float32))


init = tf.initialize_all_variables()

Loss_data = np.empty(shape=[1],dtype=float)


y_true, y_pred = None, None

ClassLabelFinal = []


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,loss],feed_dict={X:X_train,Y:y_train})
        Loss_data = np.append(Loss_data,cost)

    y_pred = sess.run(tf.argmax(yPredbyNN,1),feed_dict={X: X_test})
    y_true = sess.run(tf.argmax(y_test,1))
    print('Test accuracy: ',round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}) , 5))


    saver.save(sess, 'mymodelaudio')

    
    print "Now, Testing the unlabel data and writing the results to CSV file"
    YPredByNNForUnlabeledData = sess.run(tf.argmax(yPredbyNN,1),feed_dict={X: XFinalTest})
    print YPredByNNForUnlabeledData
    for i in xrange (len(YPredByNNForUnlabeledData)):
    	if YPredByNNForUnlabeledData[i] == 0:
    		ClassLabelFinal.append('pad')
    	else:
    		ClassLabelFinal.append('knuck')

cwd = os.getcwd()
Test_dataset_path = ("%s/data/test")%cwd
Test_dataset, Total_Instances = load_instances(Test_dataset_path)

timestamps = load_timestamps(Test_dataset)

write_results(timestamps, ClassLabelFinal, 'Result.csv')



fig = plt.figure(figsize=(10,10))
plt.title("Loss funtion vs  Training Epoch")
plt.plot(Loss_data)
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Trainig Epochs')
plt.axis([0,training_epochs,0,np.max(Loss_data)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print "F-Score: ", round(f,2)
print "Precision: ", round(p,2)
print "Recall: ", round(r,2)



