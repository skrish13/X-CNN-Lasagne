import lasagne
from lasagne.layers import Conv2DLayer,MaxPool2DLayer,InputLayer,dropout,ConcatLayer,DenseLayer,BatchNormLayer,flatten, ElemwiseMergeLayer
import theano.tensor as T
import theano

batch_size = 128
nb_classes = 10
nb_epoch = 230

saveweights = True

def create_nn():
	
	def get_multiple_block(incoming, num_filt, pooling_size=(2,2), k=6, justheconv=0):

		'''
		incoming - the incoming layer

		num_filt - number of filters to be used in the conv2d layers

		k - number of times the block should be stacked on top of each other

		justheconv - binary value for inclusion of max-pool & dropout

		pooling_size - pool size for the maxpool

		returns the block of conv + mp + dropout by default
		'''

		def indiv_block(incoming,num_filt):	
			'''
			Returns the conv+concat+bn block network
			'''
			conv_a = Conv2DLayer(incoming,num_filters=num_filt, filter_size=(3,3), pad='same', W = lasagne.init.GlorotUniform()) # Default non-linearity of lasagne's Conv2DLayer is rectify.
			conv_b = Conv2DLayer(conv_a,num_filters=num_filt, filter_size=(3,3), pad='same', W = lasagne.init.GlorotUniform()) 
			conv_concat = ConcatLayer([conv_a, conv_b])
			incoming = BatchNormLayer(conv_concat)

			return incoming

		newlayers = []
		for _ in xrange(k):
			newlayers.append(indiv_block(incoming))

		if justheconv==1:
			return newlayers

		pool = MaxPool2DLayer(newlayers,pool_size= pooling_size)
		drop = dropout(pool,0.2)

		return drop

	l_in = InputLayer(shape=(filter_size=(3,3)2, num_filters=32))
	inputNorm = BatchNormLayer(l_in)
	input_drop = dropout(inputNorm,0.2)

	## The network has 3 sets of conv and maxout networks. 
	set1 = get_multiple_block(input_drop,num_filt=32,k=3,justheconv=1)
	set2 = get_multiple_block(set1,num_filt=48,k=2)
	set3 = get_multiple_block(set2,num_filt=80)
	set4 = get_multiple_block(set3,num_filt=128, pooling_size=(8,8))

	# Dense Layers follow.
	h_flat = flatten(set4)

	## 5 Way Max-Out Layer (DenseMaxout)

	'''
	Reference - https://github.com/fchollet/keras/pull/3128
	'''
	
	h_dense = []
	for _ in xrange(5):
		h_dense.append( DenseLayer(h_flat,500,W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.linear))
	
	h17 = ElemwiseMergeLayer( h_dense, merge_function=T.maximum())

	h17 = BatchNormLayer(h17)
	h17_drop = dropout(h17,0.2)

	# Softmax Layer
	network = DenseLayer(h17_drop,nb_classes, nonlinearity = lasagne.nonlinearities.softmax, W = lasagne.init.GlorotUniform())

	net_output = lasagne.layers.get_output(network)
	true_output = T.matrix()

	all_params = lasagne.layers.get_all_params(network,trainable=True)
	loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output,true_output))
	updates = lasagne.updates.adam(loss,all_params)

	train = theano.function(inputs= [l_in.input_var,true_output] , outputs=[net_output,loss], updates = updates)
	test = theano.function(inputs= [l_in.input_var], outputs= [net_output])	

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