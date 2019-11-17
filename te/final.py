"""
to modify:

17, 21,

"""


import numpy as np
import os
import theano
from CNN import LeNetConvPoolLayer
import cv2
import glob
from logistic_sgd import LogisticRegression

#---------------------------------------------
__path__ = "~/Pictures"
resize_row = 10
resize_col  = 20
#---------------------------------------------

img_list = glob.glob(__path__ + "*.jpg")

def main(batch_size = 500 , n_epochs = 200,
         nkerns = [20, 50], learning_rate = 0.1):


    rng = numpy.random.RandomState(23455)


    x = T.matrix('x')
    y = T.ivector('y')

    layer0_input = x.reshape((batch_size, 1, 3, resize_row, resize_col))

    layer0 = LeNetConvPoolLayer(
                    rng,
                    input=layer0_input,
                    img_shape = (batch_size, 3, resize_row, resize_col),
                    filter_shape = (nkerns[0], 3, 5, 5),
                    poolsize = (2, 2))

    layer1 = LeNetConvPoolLayer(rng,
                        input=layer0.output
                        image_shape=(batch_size,nkerns[0], (resize_row - 4)/2,(resize_col-4)/2),
                        filter_shape=(nkerns[1], nkerns[1], 5, 5),
                        poolsize = (2,2))

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
                rng,
                input=layer2.input,
                n_in = nkerns[1] * number_to_be_changed*same,
                nout = 500,
                activation = T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in = 500, n_out=10)
    cost = layer3.negative_log_likelihood(y)
    
