import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from models.cnn_gru.model_utils import *

class CNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32,3,padding='same',activation='relu')
        self.conv2 = layers.Conv2D(64,3,padding='same',activation='relu')
        self.conv3 = layers.Conv2D(128,3,padding='same',activation='relu')
        self.maxpool = layers.MaxPool2D(strides=2)
        self.globalpool = layers.GlobalMaxPool2D()
    def call(self,x):
        x = self.conv3(self.maxpool(self.conv2(self.maxpool(self.conv1(x)))))
        x = self.globalpool(x)
        return x

class CNN_GRU(layers.RNN):
    def __init__(self,cnn,op_units,ip_dims,return_sequences=True,return_state=True):
        cell = CNN_GRU_Cell(cnn,op_units,ip_dims)
        super().__init__(cell,return_sequences,return_state)
    def call (self,inputs):
        return super().call(inputs)  

class Encoder(tf.keras.Model):
    def __init__(self,cnn,ip_dims):
        super().__init__()
        self.units = ip_dims
        self.cnn_gru = CNN_GRU(cnn,self.units,ip_dims,return_sequences=True,
                               return_state=True)
    def call (self,x):
        output,state = self.cnn_gru(x)
        # output -> bs,seq_len,units
        # state -> bs,units
        return output,state


class CnnGru(tf.keras.Model):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = CNN()
        ip_dims = self.cnn.compute_output_shape((None,None,None,5))[-1]
        self.encoder = Encoder(self.cnn,ip_dims)
        self.flatten = layers.Flatten()
        self.drop = layers.Dropout(0.3)
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(4)

    def call(self, x, training=False):
        x,_ = self.encoder(x)
        x = self.flatten(x)
        x = self.drop(x,training=training)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x