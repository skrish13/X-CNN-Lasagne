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
	Returns the 'X-KerasNet'

	Using default values of adam - learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

	Input to the NN is (batch_size,3,32,32) and corresponding classes it belong to (batch_size,)
	'''

	a_l_in = InputLayer((batch_size,1,32,32))
	a_l_in_bn = BatchNormLayer(a_l_in)
	b_l_in = InputLayer((batch_size,1,32,32))
	b_l_in_bn = BatchNormLayer(b_l_in)
	c_l_in = InputLayer((batch_size,1,32,32))
	c_l_in_bn = BatchNormLayer(c_l_in)

	
	a_conv1 = Conv2DLayer(a_l_in_bn,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x32x32
	b_conv1 = Conv2DLayer(b_l_in_bn,pad='same',num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx16x32x32
	c_conv1 = Conv2DLayer(c_l_in_bn,pad='same',num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx16x32x32
	
	a_conv1_1 = Conv2DLayer(a_conv1,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x32x32
	b_conv1_1 = Conv2DLayer(b_conv1,pad='same',num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx16x32x32
	c_conv1_1 = Conv2DLayer(c_conv1,pad='same',num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx16x32x32
	
	a_mp1 = MaxPool2DLayer(a_conv1_1,pool_size=(2,2)) #Bx32x16x16
	b_mp1 = MaxPool2DLayer(b_conv1_1,pool_size=(2,2)) #Bx16x16x16
	c_mp1 = MaxPool2DLayer(c_conv1_1,pool_size=(2,2)) #Bx16x16x16
	
	a_do1 = dropout(a_mp1,p=0.25) #Bx32x16x16
	b_do1 = dropout(b_mp1,p=0.25) #Bx16x16x16
	c_do1 = dropout(c_mp1,p=0.25) #Bx16x16x16

	#Exchange of feature maps

	a_to_bc = Conv2DLayer(a_do1,pad='same',num_filters=32,filter_size=(1,1),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x16x16
	b_to_a = Conv2DLayer(b_do1,pad='same',num_filters=16,filter_size=(1,1),nonlinearity=lasagne.nonlinearities.rectify)  #Bx16x16x16
	c_to_a = Conv2DLayer(c_do1,pad='same',num_filters=16,filter_size=(1,1),nonlinearity=lasagne.nonlinearities.rectify)  #Bx16x16x16

	#Merging

	a_merge1 = lasagne.layers.ConcatLayer([a_do1,b_to_a,c_to_a]) #Bx64x16x16
	b_merge1 = lasagne.layers.ConcatLayer([b_do1,a_to_bc])       #Bx48x16x16
	c_merge1 = lasagne.layers.ConcatLayer([c_do1,a_to_bc])		 #Bx48x16x16


	a_conv2 = Conv2DLayer(a_merge1,pad='same',num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx64x16x16
	b_conv2 = Conv2DLayer(b_merge1,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x16x16
	c_conv2 = Conv2DLayer(c_merge1,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x16x16
	
	a_conv2_1 = Conv2DLayer(a_conv2,pad='same',num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx64x16x16
	b_conv2_1 = Conv2DLayer(b_conv2,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x16x16
	c_conv2_1 = Conv2DLayer(c_conv2,pad='same',num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify) #Bx32x16x16
	
	a_mp2 = MaxPool2DLayer(a_conv2_1,pool_size=(2,2)) #Bx64x8x8
	b_mp2 = MaxPool2DLayer(b_conv2_1,pool_size=(2,2)) #Bx32x8x8
	c_mp2 = MaxPool2DLayer(c_conv2_1,pool_size=(2,2)) #Bx32x8x8
	
	a_do2 = dropout(a_mp2,p=0.25) #Bx64x8x8
	b_do2 = dropout(b_mp2,p=0.25) #Bx32x8x8
	c_do2 = dropout(c_mp2,p=0.25) #Bx32x8x8

	#Final Merge

	merge2 = lasagne.layers.ConcatLayer([a_do2,b_do2,c_do2]) #Bx128x8x8

	flat = flatten(merge2,2) #Bx8192
	fc = DenseLayer(flat,num_units=512,nonlinearity=lasagne.nonlinearities.rectify) #Bx512
	fc_do = dropout(fc, p=0.5) 
	network = DenseLayer(fc_do, num_units=nb_classes, nonlinearity=lasagne.nonlinearities.rectify) #Bxnb_classes

	net_output = lasagne.layers.get_output(network)
	true_output = T.matrix()

	all_params = lasagne.layers.get_all_params(network,trainable=True)
	loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output,true_output))
	updates = lasagne.updates.adam(loss,all_params)

	train = theano.function(inputs= [a_l_in.input_var,b_l_in.input_var,c_l_in.input_var,true_output] , outputs=[net_output,loss], updates = updates)
	test = theano.function(inputs= [l_in.input_var], outputs= [net_output])

	return train,test,network

def train_model(train,network):

	'''
	Write your own implementation of training the model
	'''

	# Depending on the complexity, the data can be sliced here and passed as three inputs
	# Depending on the type of partition of data, sometimes it's easier to pass it as 3 inputs seperately...
	a_train,b_train,c_train,Y_train = get_training_set()

	# Or can be seperated inside the neural network itself...
	# X_train,Y_train = get_training_set()

	for k in xrange(nb_epoch):
		net_output,loss = train(a_train,b_train,c_train,Y_train)
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

	# Depending on what input is used in the network, change the input in testing set accordingly..
	# X_test,Y_test = get_testing_set()
	a_test,b_test,c_test,Y_test = get_testing_set()

	predictions = test(a_test,b_test,c_test)
	### Test accuracy can also be calculated depending on the type of data/metrics needed using 'predictions' and 'Y_test'

	'''
	In case testing in batch size is needed, similar thing can be implemented also be implemented similarly.
	'''

	predictions = []
	for i in xrange(0,total_number,batch_size):
		a_test_batch,b_test_batch,c_test_batch = get_batch_testing_set()
		predictions += test(a_test_batch,b_test_batch,c_test_batch)
		### Test accuracy can also be calculated depending on the type of data/metrics needed using 'predictions' and 'Y_test'
	
if __name__ = '__main__':

	train,test,network = create_nn()
	train_model(train,network)
	test_model(test,network)


