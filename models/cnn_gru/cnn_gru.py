
import numpy as np
import tensorflow as tf
# import logging as mylogging
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from models.cnn_gru.model_utils import *

# logger = mylogging.getLogger('logger')

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


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
    
        # query is decoder hidden state
        # values is entire encoder hidden states
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, seq_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size,1 hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1, keepdims=True)

        return context_vector, attention_weights

    
class Decoder(tf.keras.Model):
    def __init__(self,units,emb_dim,seq_len,atten_units):
        # atten_units are intermediate dims for attention. 
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(seq_len,emb_dim)
        self.units = units
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = Attention(atten_units)
    def call(self,x,hidden,enc_output):
        x = tf.reshape(x,(-1,1))
        # x -> bs,1
        x = self.embedding(x)
        # x -> bs,emb_dim
        context,atten_w = self.attention(hidden,enc_output) 
        # context -> bs,units
        # atten_w -> bs,seq_len,1
        x = tf.concat([context,x],axis=-1)
        # x -> bs,emb_dim + units
        output,state = self.gru(x,initial_state=hidden)
        # output -> bs,1,units
        # state -> bs,units
        return tf.squeeze(output), state, atten_w
    

class CnnGru(tf.keras.Model):
    def __init__(self, seq_len, emb_dim=4, atten_units=25, final_rep=1):
        """ Arguments
            CNN -> CNN Class
            Encoder -> Encoder Class
            Decoder -> Decoder Class
            Attention -> Attention Class
            emb_dim -> embedding dimension at decoder
            seq_len -> sequence length
            atten_units -> hidden dims for Attention
            final_rep -> final output shape

            Returns
            final_op -> final ouput (bs,output_seq,final_rep)
            all_atten -> attention (bs,output_seq,seq_len,1)  

        """
        super().__init__()
        self.cnn = CNN()
        ip_dims = self.cnn.compute_output_shape((None,None,None,5))[-1]
        self.encoder = Encoder(self.cnn,ip_dims)
        self.decoder = Decoder(ip_dims,emb_dim,seq_len,atten_units)
        self.seq_len = seq_len
        self.fc = tf.keras.layers.Dense(final_rep)
        self.decoder_output_size = 4

    def call(self, x):
        # print(f'shape of x {x.shape}')
        # Reshape the decoder input for the last batch
        bs = x.shape[0]
        decoder_input = tf.convert_to_tensor(np.tile(np.arange(self.decoder_output_size),(bs,1)))
        # print(f'decoder_input shape : ', decoder_input.shape)
        # enc_output -> bs, seq_len, units
        # enc_hidden -> bs, units
        encoder_output, encoder_hidden = self.encoder(x)
        # print(f'encoder_output shape : ', encoder_output.shape)
        
        decoder_hidden = encoder_hidden
        final_op,all_atten = [],[]
        for i in range(self.decoder_output_size):
            # dec_output shape == (bs, units)
            # dec_hiddem shape == (bs, units)
            # atten_w shape == (bs, seq_len, 1)
            decoder_output,decoder_hidden,atten_w = self.decoder(decoder_input[:,i],decoder_hidden,encoder_output)
            
            final_op.append(decoder_output)
            all_atten.append(atten_w)

        final_op = tf.transpose(tf.convert_to_tensor(final_op),perm=(1,0,2))
        final_op = self.fc(final_op)
        all_atten = tf.transpose(tf.convert_to_tensor(all_atten),perm=(1,0,2,3))

        return final_op