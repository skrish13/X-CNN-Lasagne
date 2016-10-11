import theano
import theano.tensor as T
import lasagne
import cPickle
import os
from lasagne.layers import Conv2DLayer,MaxPool2DLayer,InputLayer,dropout,ConcatLayer,DenseLayer,BatchNormLayer,flatten

batch_size = 32
nb_classes = 10
nb_epoch = 200

saveweights = True


def create_nn():

	'''
	Returns the theano function - train,test 
	Returns the 'KerasNet'

	Using default values of adam - learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

	Input to the NN is (batch_size,3,32,32) and corresponding classes it belong to (batch_size,)
	'''

	l_in = InputLayer((batch_size,3,32,32))
	l_in_bn = BatchNormLayer(l_in)
	
	conv1 = Conv2DLayer(l_in_bn,pad='same',num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx64x32x32
	conv1_1 = Conv2DLayer(conv1,pad='same',num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx64x32x32
	conv1_mp = MaxPool2DLayer(conv1_1,pool_size=(2,2)) #Bx64x16x16
	conv1_do = dropout(conv1_mp,p=0.25)

	conv2 = Conv2DLayer(conv1_do,pad='same',num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx128x16x16
	conv2_1 = Conv2DLayer(conv2,pad='same',num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx128x16x16
	conv2_mp = MaxPool2DLayer(conv2_1,pool_size=(2,2)) #Bx128x8x8
	conv2_do = dropout(conv2_mp,p=0.25)

	flat = flatten(conv2_do,2) #Bx8192
	fc = DenseLayer(flat,num_units=512,nonlinearity=lasagne.nonlinearities.rectify) #Bx512
	fc_do = dropout(fc, p=0.5) 
	network = DenseLayer(fc_do, num_units=nb_classes, nonlinearity=lasagne.nonlinearities.rectify) #Bxnb_classes

	net_output = lasagne.layers.get_output(network)
	true_output = T.matrix()

	all_params = lasagne.layers.get_all_params(network,trainable=True)
	loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output,true_output))
	updates = lasagne.updates.adam(loss,all_params)

	train = theano.function(inputs= [l_in.input_var,true_output] , outputs=[net_output,loss], updates = updates)
	test = theano.function(inputs= [l_in.input_var], outputs= [net_output])

	return train,test,network

def train_model(train,network):

	'''
	Write your own implementation of training the model
	'''
	
	X_train,Y_train = get_training_set()
	for k in xrange(nb_epoch):
		net_output,loss = train(X_train,Y_train)
		weights = lasagne.layers.get_all_param_values(network)
		
		### If training accuracy needs to be calculated simultaneously it can be done so using net_output and Y_train values according to your needs.

		if saveweights:
			location = "weights/"
			if not os.path.exists(location):
				os.makedits(location)
			#Using cPickle to store the weights..
			cPickle.dump(weights,open(location+str(k)+"_epoch_weights.pkl",w)) 


	'''
	Or, In case you want to use only "batch_size" of inputs in memory
	'''
	for k in xrange(nb_epoch):

		for i in xrange(0,total_number,batch_size):
			
			X_train_batch,Y_train_batch = get_batch_training_set()
			net_output,loss = train(X_train_batch,Y_train_batch)
			### If training accuracy needs to be calculated simultaneously it can be done so using net_output and Y_train values according to your needs.

		weights = lasagne.layers.get_all_param_values(network)

		if saveweights:
			location = "weights/"
			if not os.path.exists(location):
				os.makedits(location)
			#Using cPickle to store the weights..
			cPickle.dump(weights,open(location+str(k)+"_epoch_weights.pkl",w)) 

def test_model(test,network):

	weights = cPickle.load() #insert the desired weights file
	lasagne.layers.set_all_param_values(network,weights)

	X_test,Y_test = get_testing_set()

	predictions = test(X_test)
	### Test accuracy can also be calculated depending on the type of data/metrics needed using 'predictions' and 'Y_test'

	'''
	In case testing in batch size is needed, similar thing can be implemented as above.
	'''

	predictions = []
	
	for i in xrange(0,total_number,batch_size):

		X_test_batch,Y_test_batch = get_batch_testing_set()
		predictions += test(X_test_batch)
		### Test accuracy can also be calculated depending on the type of data/metrics needed using 'predictions' and 'Y_test'
	
if __name__ = '__main__':

	train,test,network = create_nn()
	train_model(train,network)
	test_model(test,network)


