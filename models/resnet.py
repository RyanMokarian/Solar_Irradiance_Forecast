import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras,data
from tensorflow.keras import layers,models,activations


class IdentityBlock(tf.keras.Model):
    def __init__(self,filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters,3,1,'same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters,3,1,'same')
        self.bn2 = layers.BatchNormalization()
        
    def call(self,input,training=False):
        x = self.conv1(input)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x +=  input
        return tf.nn.relu(x)

class ConvBlock(tf.keras.Model):
    def __init__(self,filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters,3,2,'same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters,3,1,'same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters,1,2)
        self.bn3 = layers.BatchNormalization()

    def call(self,input,training=False):
        x = self.conv1(input)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        shortcut = self.conv3(input)
        shortcut = self.bn3(shortcut,training=training)
        x += shortcut
        return tf.nn.relu(x)

class StraightBlock(tf.keras.Model):
    def __init__(self,filters):
        super().__init__()
        self.conv = layers.Conv2D(filters,3,1,padding='same')
        self.bn = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D(2,strides=2)
    def call(self,inputs,training=False):
        x = self.conv(inputs)
        x = self.bn(x,training=training)
        x = tf.nn.relu(x)
        x = self.maxpool(x)
        return x

class BottleNeck(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(256,1)
        self.bn1 = layers.BatchNormalization()
        self.flatten  = layers.Flatten()
        self.globalmaxpool = layers.GlobalMaxPool2D()
        self.globalavgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(256)
    def call(self,input,training=False):
        x = self.conv(input)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        y = self.globalmaxpool(x)
        y = self.flatten(y)
        z = self.globalavgpool(x)
        z = self.flatten(z)
        x = tf.concat([y,z],axis=-1)
        x  = self.fc(x)
        return x

class CustomResNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.first = StraightBlock(100)
        self.convblock = ConvBlock(100)
        self.second = StraightBlock(200)
        self.identity2 = IdentityBlock(200)
        self.third = StraightBlock(200)
        self.bottleneck  = BottleNeck()  
        
    def call(self,input,training=False):
        x = self.first(input,training=training)
        x = self.convblock(x,training=training)
        x = self.second(x,training=training)
        x = self.identity2(x,training=training)
        x = self.third(x,training=training)
        x = self.bottleneck(x,training=training)
        return x