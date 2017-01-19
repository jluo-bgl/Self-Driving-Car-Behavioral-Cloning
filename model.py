import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D

from keras import backend as K
K.set_image_dim_ordering('tf')

def nvida1(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape))
    # model.add(Input(shape=(66, 200, 3)))
    # model.add(Dropout(.5))
    model.add(Convolution2D(24, 5, 5, name='conv_1', subsample=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, name='conv_2', subsample=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Convolution2D(48, 5, 5, name='conv_3', subsample=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, name='conv_4', subsample=(1, 1)))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, name='conv_5', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(1))

    return model
