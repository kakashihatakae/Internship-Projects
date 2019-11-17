import tensorflow as tf
import sys
import os
import json
from time import time
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.gridspec as gridspec

batch_size=32
Z_dim=100
trax=np.load("trax.mpy.npy")
trax=(trax.reshape([-1,28*28]))/255
tray=np.load("tray.mpy.npy")
restore=False

z=tf.placeholder(dtype=tf.float32,shape=[batch_size,100])
x=tf.placeholder(dtype=tf.float32,shape=[batch_size,28*28])

def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))

class batchnorm():
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)

'''def batchnorm(x, train=True, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
	with tf.variable_scope(name):
		return tf.contrib.layers.batch_norm(x,decay=momentum,updates_collections=None,epsilon=epsilon,scale=True,is_training=train,scope=name)'''



keep_prob = tf.placeholder(tf.float32,name="drop_prob")

def sample_Z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

def lrelu(X, leak=0.2):
	f1 = 0.5 * (1 + leak)
	f2 = 0.5 * (1 - leak)
	return f1 * X + f2 * tf.abs(X)

def plot(samples):
    samples=np.reshape(samples,[-1,28,28,1])
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

def conv(x,num_filters,kernel=3,stride=[1,2,2,1],name="conv",padding='SAME'):
	with tf.variable_scope(name):
		w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		b=tf.get_variable('b',shape=[num_filters],
			initializer=tf.constant_initializer(0.0))


		return (tf.add(tf.nn.conv2d(x, w, strides=stride, padding=padding), b))

def deconv(x,num_filters,output_shape,kernel=3,stride=[1,2,2,1],padding='SAME',name="deconv"):
	with tf.variable_scope(name):
		w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		b=tf.get_variable('b',shape=[num_filters],
			initializer=tf.constant_initializer(0.0))
		return (tf.add(tf.nn.conv2d_transpose(x, w, strides=stride, padding=padding,output_shape=output_shape), b))


def fcn(x,num_neurons,name="fcn"):
	with tf.variable_scope(name):
		w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		b=tf.get_variable('b',shape=[num_neurons],
			initializer=tf.constant_initializer(0.0))
		return tf.nn.xw_plus_b(x,weights=w,biases=b)

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def generator(z):

    with tf.variable_scope("generator"):
        gbn0=batchnorm(name="gbn0")
        z_=lrelu(gbn0(fcn(z,num_neurons=32*32,name="fcn1")))
        gbn1=batchnorm(name="gbn1")
        h0=lrelu(gbn1(fcn(z_,num_neurons=7*7*64,name="fcn2")))
        h0 = tf.reshape(h0,[-1,7,7,64])
        gbn2=batchnorm(name="gbn2")
        h1=lrelu(gbn2(deconv(h0,num_filters=32,kernel=5,output_shape=[batch_size, 14, 14, 32],name="deconv1")))
        gbn3=batchnorm(name="gbn3")
        h2=lrelu(gbn3(deconv(h1,num_filters=1,kernel=5,output_shape=[batch_size,28,28,1],name="deconv2")))
        h3=tf.reshape(h2,[-1,784])
        h4=tf.nn.sigmoid(h3)
        return h3,h4

def discriminator(image):
	with tf.variable_scope("discriminator"):
		x = tf.reshape(image,[batch_size,28,28,1])
        dbn1=batchnorm(name='dbn1')
        net =max_pool_2x2( lrelu(dbn1 (conv(x,num_filters=32,kernel=5,name="conv1"))))
        dbn2=batchnorm(name='dbn2')
        net= max_pool_2x2(lrelu(dbn2(conv(net,num_filters=32*2,kernel=5,name="conv2"))))
        net = tf.reshape(net,[batch_size,-1])
        dbn3=batchnorm(name='dbn3')
        net= lrelu(dbn3(fcn(net,num_neurons=1024,name="fcn3")))
        d_prob = tf.nn.dropout(net,keep_prob)
        dbn4=batchnorm(name='dbn4')
        logit = lrelu(dbn4(fcn(d_prob,num_neurons=1,name="fcn4")))
        prob=tf.sigmoid(logit)
        return logit,prob
gen_logit,gen_image=generator(z=z)
with tf.variable_scope("model") as scope:

    D_logit_real,D_prob_real=discriminator(x)
    scope.reuse_variables()
    D_logit_fake,D_prob_fake=discriminator(gen_image)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
vars=tf.trainable_variables()
d_vars=[var for var in vars if 'discriminator' in var.name ]
g_vars=[var for var in vars if 'generator' in var.name ]
D_solver = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(D_loss, var_list=d_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(G_loss, var_list=g_vars)
saver=tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i=0

print "done"
for digit in range(10):
    train_set = trax[list(np.where(tray==digit)[0])]
    for it in range(20000):
        g_loss=0
        d_loss=0
        if it==0 and restore==True:
            saver=tf.train.import_meta_graph('my_test_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./'))
        if (it %10 ==0):
            z_batch=sample_Z(batch_size, Z_dim)
            samples = sess.run(gen_image, feed_dict={z:z_batch,keep_prob:1.0})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        tex=train_set[:batch_size]
    #	d = sess.run(d_vars, feed_dict={x: tex, z: z_batch})
    #	print "before updating ",d[0][0]
        for batch_i in range(train_set.shape[0] // batch_size):
            tex=train_set[batch_i*(batch_size):(batch_i+1)*(batch_size)]
    	#tex = trax[np.random.choice(trax.shape[0], batch_size, replace=False)]


            _, D_loss_curr= sess.run([D_solver, D_loss], feed_dict={x: tex, z: sample_Z(batch_size, Z_dim),keep_prob:0.5})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_Z(batch_size, Z_dim),keep_prob:1.0})
    		#_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_Z(batch_size, Z_dim)})
            g_loss+=G_loss_curr
            d_loss+=D_loss_curr
        save_path=saver.save(sess,'my_test_model')
        tex=train_set[:batch_size]
    #	print "after updating ",d[0][0]

        print('Iter: {}'.format(it))
        print "G loss per batch of images :",g_loss/(trax.shape[0] // batch_size)
        print "D_loss per batch of images:",d_loss/(trax.shape[0] // batch_size)
