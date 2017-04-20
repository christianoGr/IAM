import os,sys,pickle
import numpy as np
#from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#os.chdir('/home/christos/Documents/Articulatory-Inversion/CAE/MNGUO')


myl1=pickle.load(open('train_feats_a_lot.p','rb'))
#myl2=pickle.load(open('t2.p','rb'))
myl3=pickle.load(open('test_feats_a_lot.p','rb'))
#myl4=pickle.load(open('t4.p','rb'))
myl1=np.asarray(myl1)
myl3=np.asarray(myl3)
########################
##########################################################################
#for i in range(10):
 #       plt.plot(myl3[i,:])
  #      plt.show()
# Parameters
learning_rate = 0.01
training_epochs = 500
batch_size = 250
display_step = 15
examples_to_show = 10

# Network Parameters
n_hidden_1 = 320 # 1st layer num features
n_hidden_2 = 120 # 2nd layer num features
n_hidden_3 = 12
n_input = 410 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(20000/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
	for i in range(total_batch):	        
		idx = range((i*batch_size)%20000,((i+1)*batch_size)%20000)
		
   # Use the random index to select random images and labels.
		batch_xs = myl1[idx, :]
	#batch =train_labels[idx, :]
	#dictionary={inp:x_batch,outp:y_batch}	
	#session.run(optimizer,feed_dict=dictionary)
            

            # Run optimization op (backprop) and cost op (to get loss value)
	        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    tf.train.Saver()
    #Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={X: myl3[:examples_to_show]})
    n=range(0,410)
    for i in range(examples_to_show):
        plt.plot(n,encode_decode[i,:],n,myl3[i,:])
        plt.show()
    # Compare original images with their reconstructions
    #f, a = plt.subplots(2, 10, figsize=(1, 2))
    #for i in range(examples_to_show):
     #   a[0][i].show(myl3[i,:])
	#a[1][i].imshow(encode_decode[i])
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()

