from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pickle

# set the model
input_f = Input(shape=(41,))
num_hidden = input('give num of hidden layers : ')
layers=[0]*num_hidden
lay=[]
for k in range(num_hidden):
	neurons=input('give num of neurons for layer '+str(k+1)+str(' : '))
	layers[k]=neurons
	if k==0:
		lay.append(Dense(neurons, activation='relu')(input_f))
	else:
		lay.append(Dense(neurons,activation='relu')(lay[k-1]))
artic = Dense(12, activation='sigmoid')(lay[-1])
print layers
# implement the model
deep_nn = Model(input_f, artic)
opt = raw_input('give optimizer...(sgd,adam,rmsprop,adadelta) : ')
ep = input('give num of epochs.. : ')
deep_nn.compile(optimizer=opt, loss='mean_squared_error')

data_dir = '/home/christos/Documents/Articulatory-Inversion/ArticulatoryInversion-New/'
save_dir = '/home/cpalyvos/'
print 'Preparing Data . . . '
#
train_data = data_dir+'train_1frame.txt'
train_labels = data_dir+'train_labels_1frame.txt'
valid_data = data_dir+'valid_1frame.txt'
valid_labels = data_dir+'valid_labels_1frame.txt'
test_data = data_dir+'test_1frame.txt'
#test_labels = data_dir+'test_labels._1frametxt'
trd=open(train_data,'r')
trl=open(train_labels,'r')
vald=open(valid_data,'r')
vall=open(valid_labels,'r')
ted=open(test_data,'r')
#tel=open(test_labels,'r')
traind=[]
trainl=[]
validd=[]
validl=[]
testd=[]
#testl=[]
for line1 in trd:
	l1=map(np.float32,line1.split())
	traind.append(l1)
for line1 in trl:
	l1=map(np.float32,line1.split())
	trainl.append(l1)
#for line1 in ted:
#	l1=map(np.float32,line1.split())
#	testd.append(l1)
for line1 in vald:
	l1=map(np.float32,line1.split())
	validd.append(l1)
for line1 in vall:
	l1=map(np.float32,line1.split())
	validl.append(l1)
#
x_train = np.asarray(traind)
x_valid = np.asarray(validd)
y_train = np.asarray(trainl)
y_valid = np.asarray(validl)
#x_test = np.asarray(testd)
#
deep_nn.fit(x_train, y_train,
								nb_epoch=ep,
                batch_size=256,
                shuffle=True,
								verbose=2, # message per epoch
                validation_data=(x_valid, y_valid))
weights=[]
biases=[]
for k in range(num_hidden+1):
	weights.append(deep_nn.layers[k+1].get_weights()[0])
	biases.append(deep_nn.layers[k+1].get_weights()[1])
# to save them	
f1 = open('./keras_dnn_weights.pckl', 'wb')
pickle.dump(weights, f1)
f1.close()
f2 = open('./keras_dnn_biases.pckl', 'wb')
pickle.dump(biases, f2)
f2.close()







