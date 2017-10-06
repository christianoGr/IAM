from keras.layers import Input, Dense
from keras.models import Model
from keras import initializers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
# set the model
# load biases from dnn model
f = open('./keras_dnn_biases.pckl', 'rb')
biases = pickle.load(f)
#
num_hidden=len(biases)-1
encoded_list=[len(x) for x in biases]
decoded_list=[encoded_list[len(encoded_list) -k -1] for k in range(1,len(biases))]
decoded_list.append(41)
f.close()
# load weights from dnn model
f = open('./keras_dnn_weights.pckl', 'rb')
weights = pickle.load(f)
f.close()
#input layer
input_f = Input(shape=(41,))
encoded=[]
decoded=[]
#set the encoding layers
for k in range(num_hidden+1):
	if k==0:
		encoded.append(Dense(encoded_list[k],
												kernel_initializer = initializers.constant(weights[k]),
												bias_initializer   =   initializers.constant(biases[k]),
												activation='relu')(input_f))
	else:
		encoded.append(Dense(encoded_list[k],
												kernel_initializer = initializers.constant(weights[k]),
												bias_initializer   =   initializers.constant(biases[k]),
												activation='relu')(encoded[k-1]))
	print encoded[k]
#set the decoding layers
for k in range(num_hidden+1):
	if k==0:
		decoded.append(Dense(decoded_list[k],
												kernel_initializer=initializers.constant(weights[-1*(k+1)].T),
												bias_initializer=initializers.constant(biases[-1*(k+1)]),
												activation='relu')(encoded[-1]))
	elif k==num_hidden:
		decoded.append(Dense(decoded_list[k],
												kernel_initializer=initializers.constant(weights[-1*(k+1)].T),
												bias_initializer='zeros',
												activation='sigmoid')(decoded[k-1]))
	else:
		decoded.append(Dense(decoded_list[k],
												kernel_initializer=initializers.constant(weights[-1*(k+1)].T),
												bias_initializer=initializers.constant(biases[-1*(k+1)]),
												activation='relu')(decoded[k-1]))
	print decoded[k]
# implement the model
autoencoder = Model(input_f, decoded[-1])
opt = raw_input('give optimizer...(sgd,adam,rmsprop,adadelta) : ')
ep = input('give num of epochs.. : ')
autoencoder.compile(optimizer=opt, loss='mean_squared_error')

data_dir = '/home/christos/Documents/Articulatory-Inversion/ArticulatoryInversion-New/'
save_dir = '/home/cpalyvos/'
print 'Preparing Data . . . '
#
train_data = data_dir+'train_1frame.txt'
#train_labels = data_dir+'train_labels_1frame.txt'
valid_data = data_dir+'valid_1frame.txt'
#valid_labels = data_dir+'valid_labels_1frame.txt'
test_data = data_dir+'test_1frame.txt'
#test_labels = data_dir+'test_labels._1frametxt'
trd=open(train_data,'r')
#trl=open(train_labels,'r')
vald=open(valid_data,'r')
#vall=open(valid_labels,'r')
ted=open(test_data,'r')
#tel=open(test_labels,'r')
traind=[]
#trainl=[]
validd=[]
#validl=[]
testd=[]
#testl=[]
for line1 in trd:
	l1=map(np.float32,line1.split())
	traind.append(l1)
for line1 in ted:
	l1=map(np.float32,line1.split())
	testd.append(l1)
for line1 in vald:
	l1=map(np.float32,line1.split())
	validd.append(l1)

x_train = np.asarray(traind)
x_valid = np.asarray(validd)
x_test = np.asarray(testd)
# Training the model . . .

autoencoder.fit(x_train, x_train,
								nb_epoch=ep,
                batch_size=256,
                shuffle=True,
								verbose=2, # message per epoch
             validation_data=(x_valid, x_valid))
print 'Saving the model . . .'
weights_ae=[]
biases_ae=[]
for k in range(2*num_hidden+2):
	weights_ae.append(autoencoder.layers[k+1].get_weights()[0])
	biases_ae.append(autoencoder.layers[k+1].get_weights()[1])
f = open('keras_ae_weights.pckl', 'wb')
pickle.dump(weights_ae, f)
f.close()
f = open('keras_ae_biases.pckl', 'wb')
pickle.dump(biases_ae, f)
f.close()
# to load them back 
# ---->
#f = open('store.pckl', 'rb')
#obj = pickle.load(f)
#f.close()
# <-----
print 'Printing results  . . .'
x=[]
y=[]
results = autoencoder.predict(x_test)
errors=[]
for i in range(x_test.shape[0]):
	x.append(x_test[i])
	y.append(results[i])
	#z=np.abs(x-y)
	#n=41
	#error=0
	#for j in range(n):
	#	error = error + z[j]**2
	#error=error/n
	#error=error**0.5
	#errors.append(error)
flattened_x = []
for sublist in x:
    for val in sublist:
        flattened_x.append(val)
flattened_y = []
for sublist in y:
    for val in sublist:
        flattened_y.append(val)
#first 1000 of our test data
plt.plot(flattened_x[:1000],'b',flattened_y[:1000],'r')
plt.show()

