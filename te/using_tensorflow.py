import tensorflow as tf
import cv2
import numpy as np
import random
import glob

#creating a session to run the code on
sess = tf.Session()

#storge class to store all the important variables
class storage_class(object):
    #the size of the resized images
    resize_col = 1080*0.2
    resize_row = 720*0.2

    labels = []
    batch_size = 20
    total_size = resize_row*resize_col
    single_label = 1

#function to load a list of images
def load_dataset(path, file_ext=".jpg"):
    '''
    resize the image
    assuming row*col = 720*1080
    '''

    images = []
    print ('------------------------------------------')
    print ('loading data_set')
    print ('path :' , path)
    print ('file_ext : ', file_ext)
    print ('------------------------------------------')

    # images_list -> stores a list of names
    # of images in the form of a strig
    images_list = glob.glob(path + "*" + file_ext)
    for i in images_list:
        img = cv2.imread(i)
        img = cv2.resize(img, (storage_class.resize_col, storage_class.resize_row), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        images.append(img)
    images = np.arry(images)
    return images

def set_labels(val):
    storage_class.labels = tf.ones([val, storage_class.batch_size], tf.int32)


def cnn_model_fn(mode):
    # variables that can be initialized in the sess.run()
    x = tf.placeholder(tf.float32, shape=[None, storage_class.resize_col, storage_class.resize_row, 3])
    y_label = tf.placeholder(tf.int32, shape=[None ,1])

    # returns an Output of the same dimension as that
    # of the imput , after padding it
    conv1 = tf.layers.conv2d(
            inputs = x,
            filter_size = 32,
            kernel_size = [5, 5, 3],
            padding = "same",
            activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = [2, 2],
            strides = 2
    )

    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5, 5, 32],
            padding = "same",
            activation = tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size = [2,2],
            stride = 2
    )

    #the output of the pool2 layer needs to be flattened
    # to be fed to the the fully connected layers
    pool2_flat = tf.layers.flatten(pool2)
    pool2_shape = pool2_flat.get_shape()

    # start of fully connected layer
    # used the activation function
    dense = tf.layers.dense(
             inputs=pool2_flat,
             units=pool2_shape[1],
             activation = tf.nn.sigmoid
    )

    dropout = tf.layers.dropout(
              inputs=dense,
              rate=0.4,
              training = (mode == tf.estimator.ModeKeys.TRAIN)
    )
    sess.run(tf.global_variables_initializer())

    #final layer
    logits = tf.layer.dense(inputs=dropout,
                            units=1,
                            activation=tf.nn.sigmoid)

    set_labels(1)
    
    loss = tf.losses.sigmoid_cross_entropy(
           multi_class_labels = y_label,
           logits = logits
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        k = logits>0.5

        pred = tf.equal(k, y_labels)
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    sess.run(tf.global_variables_initializer())



def main():
    pass
