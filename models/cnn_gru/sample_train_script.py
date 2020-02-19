import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras,data
from tensorflow.keras import layers,models,activations
from model_utils import *


model = Full_Model(CNN,Encoder,Decoder,BahdanauAttention)

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_loss = keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


@tf.function
def train(samples,labels):
    with tf.GradientTape() as tape:
        preds,_ = model(samples,labels,training=True)
        loss = loss_func(labels,preds)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss(loss,sample_weight=labels.shape[0])
    train_accuracy(labels,preds)
    return loss


@tf.function
def valid(samples,labels):
    preds,atten_w = model(samples,labels,training=False)
    loss = loss_func(labels,preds)
    valid_loss(loss,sample_weight=labels.shape[0])
    valid_accuracy(labels,preds)
    return loss,atten_w


epochs = 5
for i in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()

    for samples,labels in train_dataset:
        loss = train(samples,labels)
    for samples,labels in valid_dataset:
        _,atten_w = valid(samples,labels)


    print(f"Epoch: {i+1}, TLoss: {train_loss.result().numpy():.4f}, \
    TAcc: {train_accuracy.result().numpy():.4f}   , \
    VLoss: {valid_loss.result().numpy():.4f},  \
    VAcc: {valid_accuracy.result().numpy():.4f}")