import numpy as np
import pandas as pd
from keras import Sequential
from keras import layers
from keras.models import Model
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Flatten, Conv1D
from keras.layers import MaxPooling1D, GRU, Input,Masking, Concatenate, dot
from keras.optimizers import Adam, SGD
from keras.losses import MeanAbsoluteError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Attention, Concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer, RepeatVector, Add

debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))

class Attention1(object if debug_flag else Layer):
    SCORE_LUONG = 'luong'
    SCORE_BAHDANAU = 'bahdanau'

    def __init__(self, units: int = 128, score: str = 'luong', **kwargs):
        super(Attention1, self).__init__(**kwargs)
        if score not in {self.SCORE_LUONG, self.SCORE_BAHDANAU}:
            raise ValueError(f'Possible values for score are: [{self.SCORE_LUONG}] and [{self.SCORE_BAHDANAU}].')
        self.units = units
        self.score = score

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            # W in W*h_S.
            if self.score == self.SCORE_LUONG:
                self.luong_w = Dense(input_dim, use_bias=False, name='luong_w')
                # dot : last hidden state H_t and every hidden state H_s.
                self.luong_dot = Dot(axes=[1, 2], name='attention_score')
            else:
                # Dense implements the operation: output = activation(dot(input, kernel) + bias)
                self.bahdanau_v = Dense(1, use_bias=False, name='bahdanau_v')
                self.bahdanau_w1 = Dense(input_dim, use_bias=False, name='bahdanau_w1')
                self.bahdanau_w2 = Dense(input_dim, use_bias=False, name='bahdanau_w2')
                self.bahdanau_repeat = RepeatVector(input_shape[1])
                self.bahdanau_tanh = Activation('tanh', name='bahdanau_tanh')
                self.bahdanau_add = Add()

            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')

            # exp / sum(exp) -> softmax.
            self.softmax_normalizer = Activation('softmax', name='attention_weight')

            # dot : score * every hidden state H_s.
            # dot product. SUM(v1*v2). H_s = every source hidden state.
            self.dot_context = Dot(axes=[1, 1], name='context_vector')

            # [Ct; ht]
            self.concat_c_h = Concatenate(name='attention_output')

            # x -> tanh(w_c(x))
            self.w_c = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention1, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention1, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        h_s = inputs
        if debug_flag:
            self.build(h_s.shape)
        h_t = self.h_t(h_s)
        if self.score == self.SCORE_LUONG:
            # Luong's multiplicative style.
            score = self.luong_dot([h_t, self.luong_w(h_s)])
        else:
            # Bahdanau's additive style.
            self.bahdanau_w1(h_s)
            a1 = self.bahdanau_w1(h_t)
            a2 = self.bahdanau_w2(h_s)
            a1 = self.bahdanau_repeat(a1)
            score = self.bahdanau_tanh(self.bahdanau_add([a1, a2]))
            score = self.bahdanau_v(score)
            score = K.squeeze(score, axis=-1)

        alpha_s = self.softmax_normalizer(score)
        context_vector = self.dot_context([h_s, alpha_s])
        a_t = self.w_c(self.concat_c_h([context_vector, h_t]))
        return a_t

    def get_config(self):
        config = super(Attention1, self).get_config()
        config.update({'units': self.units, 'score': self.score})
        return config

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

train_rate =  0.9
seq_len = 24
output_dim = pre_len = 1
batch_size = 20
lr = 0.001
training_epoch = 490
gru_units = 64

    
data = pd.read_csv('.input.csv',header=None)
time_len = data.shape[0]
num_nodes = 1

data1 =np.mat(data,dtype=np.float32)

max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

import numpy as np

def split_and_reshape_data(trainX, trainY, testX, testY, num_splits):
    original_shape = trainX.shape
    original_shape2 = testX.shape
    train_X_split = np.split(trainX, num_splits, axis=2)
    train_Y_split = np.split(trainY, num_splits, axis=2)
    test_X_split = np.split(testX, num_splits, axis=2)
    test_Y_split = np.split(testY, num_splits, axis=2)
    split_trainX = []
    split_trainY = []
    split_testX = []
    split_testY = []
    for i in range(num_splits):
        split_trainX.append(train_X_split[i].reshape(original_shape[0], original_shape[1], 1))
        split_trainY.append(train_Y_split[i].reshape(original_shape[0], 1, 1))
        split_testX.append(test_X_split[i].reshape(original_shape2[0], original_shape[1], 1))
        split_testY.append(test_Y_split[i].reshape(original_shape2[0], 1, 1))
    return split_trainX, split_trainY, split_testX, split_testY

split_trainX, split_trainY, split_testX, split_testY = split_and_reshape_data(trainX, trainY, testX, testY, 22)
trainX22= split_trainX[21]
trainY22 =split_trainY[21]
testX22= split_testX[21]
testY22= split_testY[21]

def grumodel():
    inputs = Input(shape=(24,1))
    gru= GRU(64, return_sequences=True)(inputs)
    flattened = Flatten()(gru)
    output = Dense(1)(flattened)
    model = Model(inputs=inputs, outputs=output)
    return model

GRU_model = grumodel()
GRU_model.compile(optimizer = 'adam' , loss = 'mean_squared_error')
GRU_model.summary()

def CNN_gru_attention():
    inputs = Input(shape=(24,1))
    cnn = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    lstm = GRU(64, return_sequences=True)(cnn)
    context = Attention1(units=64, score='luong')(lstm)
    flattened = Flatten()(context)
    output = Dense(1)(flattened)
    model = Model(inputs=inputs, outputs=output)
    return model

CNN_gru_attention_model = CNN_gru_attention()
CNN_gru_attention_model.compile(optimizer = 'adam' , loss = 'mean_squared_error')
CNN_gru_attention_model.summary()

def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, r2, var

def train_and_evaluate_lstm_attention(trainX, trainY, testX, testY):
    history = GRU_model.fit(trainX, trainY, epochs=500, batch_size=500 )
    y_pred = GRU_model.predict(testX)
    testY=testY.reshape(316,1)
    rmse = mean_squared_error(testY, y_pred, squared=False)
    mae = mean_absolute_error(testY, y_pred)
    r2 = r2_score(testY, y_pred)
    var = explained_variance_score(testY, y_pred)
    rmse *= max_value
    mae *= max_value
    pd.DataFrame(y_pred * max_value).to_excel('.output.xlsx')
    return rmse, mae, r2, var
rmse, mae, r2, var = train_and_evaluate_lstm_attention(trainX22, trainY22, testX22, testY22)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)
print("Explained Variance:", var)

