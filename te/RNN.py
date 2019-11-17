import numpy as np
import tensorflow as tf

'''
class RNN(object):
    """docstring for RNN."""
    def __init__(self, input_from_cnn, output_size, hidden_layer_size, labels, ):
        #assert input_from_cnn.shape == input_size
        self.output_size = output_size

        #input_size = batchsize*inputs_size
        self.input_size = input_from_cnn.shape
        self.input_cnn = input_from_cnn
        self.hidden_layer_size = input_layer_size
        self.labels = labels

        #Wxh = [256, 256] == [hidden.size, input.size]
        self.Wxh = np.random.rand(self.hidden_layer_size, self.input_size[1])

        #Whh = [256, 256] == [hidden.size, hidden.size]
        self.Whh = np.random.rand(self.hidden_layer_size, self.hidden_layer_size)

        #Why = [10, 256] == [output_size, hidden.size]
        self.Why = np.random.rand(self.output_size, self.output_size)

        #bh = [hidden ,1] , by = [output, 1]
        bh = np.random.rand(self.hidden_layer_size, 1)
        by = np.random.rand(self.output_size, 1)

        #y
        self.y = np.array([])

        #ht_prev
        self.ht_prev = np.random.rand(self.hidden_layer_size, 1)

        self.ht_full = np.array([])
        self.ht_prev_full = np.array([])

        # NOTE: check for the correctness of rnn_output
        def rnn_output(self):
            resetHs()
            for i in self.input_cnn:
                ht = np.tanh(np.sum(np.dot(self.Wxh, i) + np.dot(Whh, self.ht_prev) , bh))
                np.concatenate((self.ht_full, ht), axis = 1)

                y_temp = np.sum(tf.dot(self.Why, ht) , by)
                np.concatenate((self.ht_prev_full, self.ht_prev), axis = 1)

                self.ht_prev = ht
                np.concatenate((self.y, y_temp), axis = 1)

        # NOTE: check np.sum and np.square in numpy
        def Least_squares(self):
            loss_list = []
            loss = []
            for i_y, i_labels in self.y, self.labels:
                loss_list.append([np.sum(np.square(i_y - i_labels))])
            loss_batch = np.sum(loss_list)
            return loss_list, loss_batch

        def BPTT(self):
            grads = grad_least_squares(self.labels, self.y)
            dWhy = np.dot(grads, np.transpose(self.ht_full))

            dhprev = np.zeros_like(self.ht_prev)
            for i_ht_prev, i_ht, i_x in self.ht_prev_full , self.ht_full, self.input_cnn:
                d_whh = i_ht_prev + dhprev
                d_wxh = i_x + dhprev
                dtan = (1 - i_ht**2)
                dWhh += np.dot(dtan*d_Whh, np.dot(np.transpose(Why), grads))
                dWxh += np.dot(dtan*d_Wxh, np.dot(np.transpose(Why), grads)))
                dhprev = np.dot(np.transpose(Why),dtan)
            return dWhy, dWxh, dWhh

        def grad_least_squares(self, label, pred):
            return 2*(label-pred)

        def resetHs(self):
            self.ht_full = np.array([])
            self.ht_prev_full = np.array([])

        def LSTM(self):
            input_cnn = tf.placeholder(tf.float32,[self.input_size[0], self.input_size[1]])
            lstm_cells = tf.contrib.rnn.BasicLSTMCell(self.input_size[1])

            for input_ in input_cnn:
                outout, state = lstm_cells()
'''
batchsize = 11
input_size = 10
no_lstm_units = 12
learning_rate = 0.001

labels = tf.placeholder(tf.float32, shape=[batchsize, input_size])
input_from_cnn = tf.placeholder(tf.float32, shape=[batchsize, input_size])
W = tf.Variable(np.random.rand(no_lstm_units, input_size), dtype = tf.float32)
b = tf.Variable(np.random.rand(1,input_size), dtype = tf.float32)

lstm_input = tf.placeholder(tf.float32, [input_from_cnn.shape[0], input_from_cnn.shape[1]])
LSTM_layer = tf.contrib.rnn.BasicLSTMCell(no_lstm_units)
output, states = tf.nn.static_rnn(LSTM_layer, [input_from_cnn], dtype = tf.float32)

logits = tf.matmul(output[-1],W) + b
loss = tf.losses.mean_squared_error(labels = labels, predictions = logits)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

print "----------------------------------------"
print " initializing variables"
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print "done"
print "----------------------------------------"

dummy_input = np.random.rand(batchsize, input_size)
dummy_labels = np.random.rand(batchsize, input_size)

#print dummy_input
#print dummy_labels
#print output[-1]
feed_dict = {input_from_cnn:dummy_input, labels:dummy_labels}
print sess.run(loss, feed_dict = feed_dict)
