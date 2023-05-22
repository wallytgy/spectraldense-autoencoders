import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Activation, Dropout
from keras import backend as K
from tensorflow.keras.optimizers import Adam,SGD

import tensorflow as tf
from keras.models import Model
from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    ScaledL2NormPooling2D,
    FrobeniusDense,
)
from deel.lip.model import Sequential
from deel.lip.activations import GroupSort, FullSort
from tensorflow.keras.constraints import max_norm


class Encoder(keras.layers.Layer):

    def __init__(self, units = 50, reduced_dim = 6, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.units = units
        self.reduced_dim = reduced_dim
        self.W1 = SpectralDense(self.units, activation = GroupSort(2),use_bias=True, kernel_initializer="orthogonal")
        self.W2 = SpectralDense(self.reduced_dim, activation = GroupSort(2), use_bias=True, kernel_initializer="orthogonal")

    @property
    def state_size(self):
        return self.units


    def call(self, inputs):
        return self.W2(self.W1(inputs))
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config['units'] = self.units
        config['reduced_dim'] = self.reduced_dim
        return config


class DenseEncoder(keras.layers.Layer):

    def __init__(self, units = 50, reduced_dim = 6, **kwargs):
        super(DenseEncoder, self).__init__(**kwargs)
        self.units = units
        self.reduced_dim = reduced_dim
        self.W1 = Dense(self.units, activation = 'relu',use_bias=True)
        self.W2 = Dense(self.reduced_dim, activation = 'relu', use_bias=True)

    @property
    def state_size(self):
        return self.units


    def call(self, inputs):
        return self.W2(self.W1(inputs))
    
    def get_config(self):
        config = super(DenseEncoder, self).get_config()
        config['units'] = self.units
        config['reduced_dim'] = self.reduced_dim
        return config
class Decoder(keras.layers.Layer):

    def __init__(self, units = 50, output_dim = 46, maxnorm = 1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.maxnorm = maxnorm
        self.W1 = SpectralDense(self.units, activation = GroupSort(2), use_bias=True, kernel_initializer="orthogonal")
        self.W2 =Dense(self.output_dim, activation = "linear", use_bias=True, kernel_constraint=max_norm(maxnorm))

    @property
    def state_size(self):
        return self.units


    def call(self, inputs):
        return self.W2(self.W1(inputs))
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config['units'] = self.units
        config['output_dim'] = self.output_dim
        config['maxnorm'] = self.maxnorm
        return config

class DenseDecoder(keras.layers.Layer):

    def __init__(self, units = 50, output_dim = 46, maxnorm = 1, **kwargs):
        super(DenseDecoder, self).__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.maxnorm = maxnorm
        self.W1 = Dense(self.units, activation = 'relu', use_bias=True)
        self.W2 = Dense(self.output_dim, activation = "linear", use_bias=True, kernel_constraint=max_norm(maxnorm))

    @property
    def state_size(self):
        return self.units


    def call(self, inputs):
        return self.W2(self.W1(inputs))
    
    def get_config(self):
        config = super(DenseDecoder, self).get_config()
        config['units'] = self.units
        config['output_dim'] = self.output_dim
        config['maxnorm'] = self.maxnorm
        return config
