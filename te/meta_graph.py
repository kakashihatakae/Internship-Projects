import tensorflow as tf
import numpy as np

class storage_class(object):
    #the size of the resized images
    resize_factor = 0.1
    resize_col = int(1080*resize_factor)
    resize_row = int(720*resize_factor)
    labels = []
    batch_size = 20
    total_size = resize_row*resize_col
    single_label = 1

#make a session object
print "starting session"
sess = tf.Session()


print "starting program"
x = tf.placeholder(tf.float32, shape=[None, storage_class.resize_row, storage_class.resize_col, 3], name = "x")
y_label = tf.placeholder(tf.int32, shape=[None ,1], name = "y_label")
training = tf.placeholder(tf.bool, name = "training")

# returns an Output of the same dimension as that
# of the imput , after padding it
conv1 = tf.layers.conv2d(
        inputs = x,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
)
print "done conv1"

pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2, 2],
        strides = 2
)
print "done pool1"

conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
)
print "done conv2"

pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size = [2,2],
        strides = 2
)
print "done pool2"

#the output of the pool2 layer needs to be flattened
# to be fed to the the fully connected layers
pool2_flat = tf.layers.Flatten()(pool2)
pool2_shape = pool2_flat.get_shape()
print "flatten"

# start of fully connected layer
# used the activation function
dense = tf.layers.dense(
        inputs=pool2_flat,
        units=pool2_shape[1],
        activation = tf.nn.sigmoid
)
print "done dense"
'''
dropout = tf.layers.dropout(
         inputs=dense,
         rate=0.4,
         training = training
)
'''
print "done dropout"
#sess.run(tf.global_variables_initializer())
print "initialize"
#final layer
logits = tf.layers.dense(inputs=dense,
                        units= 1,
                        activation=tf.nn.sigmoid,
                        name = "logits")

tf.add_to_collection("logits", logits)

print "done logits"
loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels = y_label,
        logits = logits
)
print "done loss function"
loss_mean = tf.reduce_mean(loss)
#sess.run(tf.global_variables_initializer())
print "starting gradient descent"
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001,
                                    name="GradientDescent").minimize(loss_mean)
init = tf.initialize_all_variables()
print "end gradient descent"

tf.add_to_collection("GradientDescent", optimizer)
print "done all layers done"


sess.run(init)

print "done init"
#x_ = np.zeros([storage_class.batch_size, storage_class.resize_row, storage_class.resize_col, 3], np.float32)
#y_ = np.zeros([storage_class.batch_size ,1], np.float32)
#training_ = True


#sess.run(optimizer, {x: x_, y_label: y_ , training: training_})
print "starting saver"
saver = tf.train.Saver()
saver.save(sess, "temp/graph")
saver.export_meta_graph("temp/graph")


print "all done bitch!!!"
