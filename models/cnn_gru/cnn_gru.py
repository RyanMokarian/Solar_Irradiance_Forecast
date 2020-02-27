import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations

from models.cnn_gru.cnn import CNN, Encoder

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