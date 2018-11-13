import sys
import copy
import pickle
import csv
import os
import numpy as np
import pandas
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

PATH_TO_SOURCES = '/home/cyakaboski/src/'
MODULE_FOLDER_NAMES = ['Design-of-experiment-Python']

for module in MODULE_FOLDER_NAMES:
    mod_path = os.path.join(PATH_TO_SOURCES, module)
    if not sys.path.__contains__(mod_path):
        sys.path.append(mod_path)
    
import DOE_functions
import seaborn as sns


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1_path = 'data/cifar-10-batches-py/data_batch_1'
dict_1 = unpickle(batch_1_path)

x_train = copy.deepcopy(dict_1[b'data'])/255.0
y_train = np.array(copy.deepcopy(dict_1[b'labels']))

x_train = x_train.reshape(10000, 32, 32, 3)

test_batch_path = 'data/cifar-10-batches-py/test_batch'
dict_test = unpickle(test_batch_path)
x_test = copy.deepcopy(dict_test[b'data'])/255.0
y_test = np.array(copy.deepcopy(dict_test[b'labels']))
x_test = x_test.reshape(10000, 32, 32, 3)

dict_values = {'conv_filters1': [10,80],
              'conv_kernelSize1': [2,10],
              'conv_strides1': [1,10],
              'conv_activation1': [1,2,3],
              'conv_filters2': [10,80],
              'conv_kernelSize2': [2,10],
              'conv_strides2': [1,10],
              'conv_activation2': [1,2,3],
              'conv_filters3': [10,80],
              'conv_kernelSize3': [2,10],
              'conv_strides3': [1,10],
              'conv_activation3': [1,2,3],
              'pool_type1': [1,2,3],
              'pool_size1': [2,10],
              'pool_strides1': [1, 10],
              'pool_momentum1': [.5, .95],
              'pool_type2': [1,2,3],
              'pool_size2': [2,10],
              'pool_strides2': [1, 10],
              'pool_momentum2': [.5, .95],
              'drop_value1': [0, .5],
              'drop_value2': [0, .5],
              'drop_value3': [0, .5],
              'dense_num' : [100, 1000]}


doe = DOE_functions.build_central_composite(dict_values, face='ccf')

doe.to_csv("doe_design-cc.csv")

act_map = {1:'sigmoid', 2:'relu', 3:'tanh'}
batch_size = 100
epochs = 20
results = []

def pool_layer(t, size, stride, momentum):
    if t == 1:
        _pool_layer = MaxPool2D(padding='Same',
                               strides= (round(stride), round(stride)),
                               pool_size=(round(size), round(size)))
    elif t == 2:
        _pool_layer = AveragePooling2D(padding='Same',
                               strides= (round(stride), round(stride)),
                               pool_size=(round(size), round(size)))
    else:
        _pool_layer = BatchNormalization(momentum=momentum)
    return _pool_layer

def conv_layer(conv_filter, activation, kernel_size, conv_stride, input_shape=None):
    if input_shape != None:
        _conv_layer = Conv2D(round(conv_filter),
                         input_shape = x_train[0].shape,
                         activation= act_map[activation],
                         kernel_size = (round(kernel_size), round(kernel_size)),
                         padding='Same',
                         strides = (round(conv_stride), round(conv_stride)))
    else:
        _conv_layer = Conv2D(round(conv_filter),
                         activation= act_map[activition],
                         kernel_size = (round(kernel_size), round(kernel_size)),
                         padding='Same',
                         strides = (round(conv_stride), round(conv_stride)))
    return _conv_layer



for test in range(0, doe.shape[0]):
    print('test:', test)
    try:
        row = doe.loc[test,:]

        model = Sequential()
        model.add(conv_layer(row[0], row[3], row[1], row[2], input_shape=x_train[0].shape))
        model.add(pool_layer(row[12], row[13], row[14], row[15]))
        model.add(Dropout(row[20]))
        model.add(conv_layer(row[4], row[7], row[5], row[6]))
        model.add(pool_layer(row[16], row[17], row[18], row[19]))
        model.add(Dropout(row[21]))
        model.add(conv_layer(row[8], row[11], row[9], row[10]))
        model.add(Flatten())
        model.add(Dense(int(row[23]), activation='relu'))
        model.add(Dropout(row[22]))
        model.add(Dense(10, activation='softmax'))

        #model.summary()

        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train, y_train, 
                                batch_size=batch_size, 
                                epochs=epochs,
                                verbose=0)
        result_loss, result_acc = model.evaluate(x_test, y_test)

        results.append((result_loss, result_acc))
    except Exception as e:
        results.append((e.message))
        continue

with open('results-cc-full', 'wb') as f:
    pickle.dump(results, f)
