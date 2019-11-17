import tensorflow as tf
import numpy as np
import cv2

img = cv2.imread("1.png")
y = [[1]]
y = np.array(y)
res = cv2.resize(img,(1080,720), interpolation = cv2.INTER_LINEAR)
img = np.array([res], dtype=np.float32)
print img.shape

sess = tf.Session()

def init_weights(shape):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)
	
W1 = init_weights([5,5,3,32])
b1 = init_weights([32])
out1 = tf.nn.leaky_relu(tf.nn.conv2d(img, W1, 
									 strides=[1,2,2,1], 
									 padding='SAME') + b1)

pool1 = tf.nn.max_pool(out1, ksize=[1,2,2,1], 
                          strides=[1,2,2,1],
                          padding='SAME')



w2 = init_weights([5,5,32,64])
b2 = init_weights([64])
out2 = tf.nn.leaky_relu(tf.nn.conv2d(pool1, w2,
									 strides=[1,2,2,1],
									 padding='SAME')+b2)
pool2 = tf.nn.max_pool(out2, ksize = [1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')
print pool2.shape

pool2_flat = tf.reshape(pool2, [-1, 45*68*64])
W_fc_1 = init_weights([45*68*64, 1000])
b_fc_1 = init_weights([1000])
FC1 = tf.nn.leaky_relu(tf.matmul(pool2_flat, W_fc_1) + b_fc_1)

W_fc_2 = init_weights([1000,1])
b_fc_2 = init_weights([1])
logits = tf.nn.leaky_relu(tf.matmul(FC1,W_fc_2) + b_fc_2)

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,
                                        logits=logits)
loss_mean = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss_mean)

init = tf.global_variables_initializer()
sess.run(init)

