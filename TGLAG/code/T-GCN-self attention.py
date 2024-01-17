import numpy as np
import pandas as pd
import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv('.adj.csv',header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv('.input.csv',header=None)
    return sz_tf, adj

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
    
import tensorflow as tf
import scipy.sparse as sp
import numpy as np

def normalized_adj(adj):
   adj = sp.coo_matrix(adj)
   rowsum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(rowsum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
   normalized_adj = normalized_adj.astype(np.float32)
   return normalized_adj
   
def sparse_to_tuple(mx):
   mx = mx.tocoo()
   coords = np.vstack((mx.row, mx.col)).transpose()
   L = tf.SparseTensor(coords, mx.data, mx.shape)
   return tf.sparse_reorder(L) 
   
def calculate_laplacian(adj, lambda_max=1):  
   adj = normalized_adj(adj + sp.eye(adj.shape[0]))
   adj = sp.csr_matrix(adj)
   adj = adj.astype(np.float32)
   return sparse_to_tuple(adj)
   
def weight_variable_glorot(input_dim, output_dim, name=""):
   init_range = np.sqrt(6.0 / (input_dim + output_dim))
   initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                           maxval=init_range, dtype=tf.float32)

   return tf.Variable(initial,name=name)  

import matplotlib.pyplot as plt

def plot_result(test_result,test_label1,path):
    ##all test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all.jpg')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    plt.plot(a_pred,'r-',label="prediction")
    plt.plot(a_true,'b-',label="true")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday.jpg')
    plt.show()
    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn.rnn_cell import RNNCell

class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))


    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):  
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        ## concat
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])  
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
#                print(x1)
            x = tf.reshape(x1, shape=[self._nodes, input_size,-1])
            x = tf.transpose(x,perm=[2,0,1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.keras.initializers.glorot_normal())
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn.rnn_cell import RNNCell
from tensorflow.python.platform import tf_logging as logging

class GRUCell(RNNCell):
    """Gated Recurrent Units. """
    
    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        """Gated Recurrent Units."""
        
        super(GRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._act = act
        self._num_nodes = num_nodes
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or "gru"):
            with tf.variable_scope("gates"): 
                # Reset gate and update gate.
                value = tf.nn.sigmoid(
                    self._linear(inputs, state, 2 * self._num_units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                c = self._linear(inputs, r * state, self._num_units, scope=scope)
#                c = self._linear(inputs, r * state, scope=scope)
                if self._act is not None:
                    c = self._act(c)
            new_h = u * state + (1 - u) * c
        return new_h, new_h
        
    
    def _linear(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
#        print('1',inputs.get_shape())
        inputs = tf.expand_dims(inputs, 2)
#        print('2',inputs.get_shape())
       
        ## state:(batch,num_node,gru_units)
#        print('2',state.get_shape())
        state = tf.reshape(state, (-1, self._num_nodes, self._num_units))
#        print('2',state.get_shape())
        
        x_h  = tf.concat([inputs, state], axis=2)
        input_size = x_h.get_shape()[2].value
        
        x = tf.reshape(x_h, shape=[-1, input_size])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias))

            x = tf.matmul(x, weights)  # (batch_size * self.num_nodes, output_size)          
            x = tf.nn.bias_add(x, biases)

            x = tf.reshape(x, shape=[-1, self._num_nodes ,output_size])
            x = tf.reshape(x, shape=[-1, self._num_nodes * output_size])
        return x  

# -*- coding: utf-8 -*-

import pickle as pkl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.flags.DEFINE_integer('embedding_dim', 128, 'Dimensionality of character embedding (default: 128)')
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
time_start = time.time()

###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 500, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 24, 'time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 20, 'batch size.')
flags.DEFINE_string('dataset', 'sz', 'sz.')
flags.DEFINE_string('model_name', 'TGCN', 'TGCN.')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
if data_name == 'sz':
    data, adj = load_sz_data('sz')
if data_name == 'los':
    data, adj = load_los_data('los')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)

max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TGCN(_X, weights, biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[seq_len,-1,num_nodes,gru_units])
    out = tf.transpose(out, perm=[1,0,2,3])

    last_output,alpha = self_attention1(out, weight_att,bias_att)

    output = tf.reshape(last_output,shape=[-1,seq_len])
    output = tf.matmul(output, weights['out']) + biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])

    return output, outputs, states, alpha

def self_attention1(x, weight_att,bias_att):
    x = tf.matmul(tf.reshape(x,[-1,gru_units]),weight_att['w1']) + bias_att['b1']
#    f = tf.layers.conv2d(x, ch // 8, kernel_size=1, kernel_initializer=tf.variance_scaling_initializer())
    f = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    g = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    h = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']

    f1 = tf.reshape(f, [-1,seq_len])
    g1 = tf.reshape(g, [-1,seq_len])
    h1 = tf.reshape(h, [-1,seq_len])
    s = g1 * f1
    print('s',s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print('bata',beta)
    context = tf.expand_dims(beta,2) * tf.reshape(x,[-1,seq_len,num_nodes])

    context = tf.transpose(context,perm=[0,2,1])
    print('context', context)
    return context, beta 
def self_attention(x, ch, weight_att, bias_att):
    f = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w'])
    g = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w']) + bias_att['b_att']
    h = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w']) + bias_att['b_att']
    print('h',h)

    f = tf.reshape(f, [-1,num_nodes])
    g = tf.reshape(g, [-1,num_nodes])
    h = tf.reshape(h, [-1,num_nodes])    
    s = g * f
    print('s',s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print('bata',beta)
#    o = tf.matmul(beta, h) # [bs, N, C]
    o = beta * h
    print('o',o)
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

#    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
#    o = tf.reshape(o, shape=x.shape)
    o = tf.expand_dims(o, 2)
    x = gamma * o + x
    print('x',x)
#    x = tf.reduce_sum(x, 2)
    return x, beta    
###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# weights
weights = {
    'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight_o')}
bias = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}
weight_att={
    'w1':tf.Variable(tf.random_normal([gru_units,1], stddev=0.1),name='att_w1'),
    'w2':tf.Variable(tf.random_normal([num_nodes,1], stddev=0.1),name='att_w2')}
bias_att = {
    'b1': tf.Variable(tf.random_normal([1]),name='att_b1'),
    'b2': tf.Variable(tf.random_normal([1]),name='att_b2')}

pred,ttto,ttts,alpha = TGCN(inputs, weights, bias)    
y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

#out = 'out/%s'%(model_name)
out = 'out/%s'%(model_name)
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 
    return batch_s
    
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
  
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output, alpha1 = sess.run([optimizer, loss, error, y_pred, alpha],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)

    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))
    
        
time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
#x = [i for i in range(training_epoch)]
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
#test_rmse = [float(i) for i in test_rmse]
var = pd.DataFrame(batch_loss1)
var.to_csv(path+'/batch_loss.csv',index = False,header = False)
var = pd.DataFrame(train_loss)
var.to_csv(path+'/train_loss.csv',index = False,header = False)
var = pd.DataFrame(batch_rmse1)
var.to_csv(path+'/batch_rmse.csv',index = False,header = False)
var = pd.DataFrame(train_rmse)
var.to_csv(path+'/train_rmse.csv',index = False,header = False)
var = pd.DataFrame(test_loss)
var.to_csv(path+'/test_loss.csv',index = False,header = False)
var = pd.DataFrame(test_acc)
var.to_csv(path+'/test_acc.csv',index = False,header = False)
var = pd.DataFrame(test_rmse)
var.to_csv(path+'/test_rmse.csv',index = False,header = False)


index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv',index = False,header = False)
plot_result(test_result,test_label1,path)
plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

fig1 = plt.figure(figsize=(7,3))
ax1 = fig1.add_subplot(1,1,1)
plt.plot(np.sum(alpha1,0))
plt.savefig(path+'/alpha.jpg',dpi=500)
plt.show()


plt.imshow(np.mat(np.sum(alpha1,0)))
plt.savefig(path+'/alpha11.jpg',dpi=500)
plt.show()

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])

