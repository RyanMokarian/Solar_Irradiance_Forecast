import tensorflow as tf
import numpy as np
from numpy import moveaxis
from numpy import asarray
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50

class DummyModel(tf.keras.Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1, activation=None)
        
    def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        return self.dense2(x)

class SunsetModel(tf.keras.Model):
    def __init__(self):
        super(SunsetModel, self).__init__()
        self.conv1 = layers.Conv2D(12, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(24, (3, 3), activation='relu', padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.maxpooling = layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense3 = layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = inputs[:,-1,:,:,:] # Only consider T0

        # Conv block 1
        x = self.conv1(x)
        #x = self.batch_norm1(x,training=training)
        x = self.maxpooling(x)
        
        # Conv block 2
        x = self.conv2(x)
        #x = self.batch_norm2(x,training=training)
        x = self.maxpooling(x)
        
        # Fully connected
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return tf.tile(input=x, multiples=tf.constant([1, 4])) # Return same prediction for T0, T+1, T+3 and T+6

class Sunset3DModel(tf.keras.Model):
    def __init__(self):
        super(Sunset3DModel, self).__init__()
        self.conv1 = layers.Conv3D(12, (3, 3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')
        self.maxpooling = layers.MaxPool3D(pool_size=(2, 2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense3 = layers.Dense(4, activation=None)

    def call(self, inputs):
    
        # Conv block 1
        x = self.conv1(inputs)
        #x = self.batch_norm(x)
        x = self.maxpooling(x)
        
        # Conv block 2
        x = self.conv2(x)
        #x = self.batch_norm(x)
        x = self.maxpooling(x)
        
        # Fully connected
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
class ConvDemModel(tf.keras.Model):
    def __init__(self):
        super(ConvDemModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.maxpooling = layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.droppedout = layers.Dropout(0.2)
        self.dense = layers.Dense(1, activation=None)

    def call(self, inputs):
        # Conv block 1:        
        x = self.conv1(inputs)   # 30x30x5 input convs to 30x30x32
        x = self.maxpooling(x)   # 30x30x32 maxpools to 15x15x32

        # Conv block 2
        x = self.conv2(x)        # 15x15x32 convs to 15x15x32
        x = self.maxpooling(x)   # 15x15x32 maxpools to 7x7x64
        
        # Conv block 3        
        x = self.conv3(x)        # 7x7x64 convs to 7x7x64
        x = self.maxpooling(x)   # 7x7x64 maxpools to 3x3x64
        
        # Conv block 4        
        x = self.conv3(x)        # 3x3x64 convs to 3x3x64        

        # Flatten, dropped out (%20) & output
        x = self.flatten(x)      
        x = self.droppedout(x)
        x = self.dense(x)

        return x

class ConvLSTM(tf.keras.Model):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.average_pool = layers.TimeDistributed(layers.AveragePooling2D(pool_size=(4, 4)))
        self.conv1 = layers.ConvLSTM2D(128, (5, 5), return_sequences = True)
        self.conv2 = layers.ConvLSTM2D(64, (5, 5), return_sequences = True)
        self.conv3 = layers.ConvLSTM2D(64, (5, 5))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense3 = layers.Dense(4, activation=None)
    
    def call(self, inputs):
        x = self.average_pool(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class ResNetModel(tf.keras.Model):
    def __init__(self, resnet_weights_path='pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        super(ResNetModel, self).__init__()
        self.resnet50 = ResNet50(include_top = False, weights = resnet_weights_path)
        self.new_resnet = ResNet50(weights=None, input_shape=(32, 32, 5), include_top=False)
        self.flatten = tf.keras.layers.Flatten()
        self.droppedout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def build(self, input_shape):
        for i, (new_layer, layer) in enumerate(zip(self.new_resnet.layers[1:], self.resnet50.layers[1:])):
            if i == 1:
                new_weights = np.zeros((7, 7, 5, 64))
                original_weights = np.array(layer.get_weights())
                new_weights[:, :, 0:3, :] = original_weights[0][:, :, :3, :]
                new_weights[:, :, 3:5, :] = original_weights[0][:, :, :2, :]
                new_layer.set_weights([new_weights,original_weights[1]])
                continue
            new_layer.set_weights(layer.get_weights())
        
    def call(self, inputs): 
        x = inputs[:,-1,:,:,:] # Only consider T0
        
        x = self.new_resnet(x)
        x = self.flatten(x)      
        x = self.droppedout(x)
        x = self.dense(x)
        return x    