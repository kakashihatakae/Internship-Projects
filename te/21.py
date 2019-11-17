import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import pool
import cv2
from Lenet import LeNetConvPoolLayer
from Lenet import HiddenLayer


class storage_class(object):
	cols = 1080
	row = 720
	resize_factor = 0.1
	resize_col = cols*resize_factor
	resize_row = row*resize_factor
	batch_size = 1
	
	

def leaky_relu(X, leak=0.2):
	f1 = 0.5 * (1 + leak)
	f2 = 0.5 * (1 - leak)
	return f1 * X + f2 * np.absolute(X)	


'''
y = [[1]]*10
y = np.array(y)	

W_filter = np.random.rand(32, 5, 5 ,1)
l1 = T.nnet.conv2d(input = img, filters = W_filter)   #(68 , 104)
l1a = pool.pool_2d(l1, (2,2))                 # 32X34X72
activation1 = leaky_relu(l1a)

W_filter2 = np.random.rand(32, 5, 5 , 1)                      #64, 100, 32
l2 = T.nnet.conv2d(input=activation1, filters=W_filter2)
l2a = pool.pool_2d(l2, (2,2))                              #32, 50, 32
activation2 = leaky_relu(l2a)

k = 32*50*32
W_fc_1 = np.random.rand(k, k)
reshaped_activation = np.reshape(activation, (32*50*32))
activated_layer1_fc = sigmoid(T.dot(reshaped_activation, W_fc_1)) #batch_size * 32*50*32

W_fc_2 = np.random.rand(32*50*32)
logits = sigmoid(T.dot(activated_layer_fc1, W_fc_2))

cost_function = T.nnet.sigmoid_binary_crossentropy(logits, y).mean()

layer = np.concatenate(W_filter, W_filter2)
layer = np.concatenate(layer, W_fc_1)
layer = np.concatenate(layer, W_fc_2)

grads = T.grad(cost_function, layer)
'''

img = cv2.imread("1.png")
img = np.array([img])
rng = np.random.RandomState(23455)

layer0 = LeNetConvPoolLayer(rng,
							input1=img,
							image_shape=(storage_class.batch_size, 3, storage_class.resize_row, storage_class.resize_col),
							filter_shape=(32, 3, 5, 5),
							poolsize=(2,2))


layer1 = LeNetConvPoolLayer(rng,
  							input1=layer0.output,
  							image_shape=(32, 32, storage_class.resize_row/2, storage_class.resize_col/2),
  							filter_shape=(32,32,5,5),
  							poolsize=(2,2))

layer2_input = layer1.output.flatten(2)	
layer2 = HiddenLayer(rng,
                     input2=layer2_input,
                     n_in = 32*18*27,
                     n_out = 32*18*27,
                     activation = leaky_relu)
                     
layer3 = HiddenLayer(rng,
                     input2=layer2.output,
                     n_in = 32*18*27,
                     n_out = storage_class.batch_size,
                     activation=T.nnet.relu)
                            
                            
W = theano.shared(value=np.zeros((storage_class.batch_size, storage_class.batch_size), dtype=theano.config.floatX), borrow=True)
b = theano.shared(value=np.zeros((storage_class.batch_size,),dtype=theano.config.floatX), borrow=True)


Wx_b = T.dot(layer3.output, W) + b
k = T.nnet.sigmoid(Wx_b)
y = [[1]]*storage_class.batch_size
y = np.array(y)
y = theano.shared(value=y, borrow=True)

print k
print y

cost = layer3.negative_log_likelihood(k,y)
params = layer0.params + layer1.params + layer2.params + layer3.params 
grads = T.grad(cost, params)
updates=[(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params, grads)]

train_model = theano.function(cost, updates=updates, givens={x:img, y:y})
