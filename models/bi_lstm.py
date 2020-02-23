import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from models.resnet import CustomResNet

class Encoder(tf.keras.Model):
    def __init__(self,ip_dims):
        super().__init__()
        self.units = ip_dims
        self.lstm = layers.Bidirectional(layers.LSTM(self.units,return_sequences=True, return_state=True),merge_mode='concat')
    def call (self,x):
        seq_output  = self.lstm(x)
        #,hidden,cell_state
        # output -> bs,seq_len,units
        # state -> bs,units
        return seq_output#,hidden,cell_state   


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
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
        self.lstm = tf.keras.layers.LSTM(self.units,return_sequences=True,return_state=True)
        self.attention = BahdanauAttention(atten_units)
    def call(self,x,hidden,cell_state,enc_output):
        x = tf.reshape(x,(-1,1))
        # x -> bs,1
        x = self.embedding(x)
        # x -> bs,emb_dim
        context,atten_w = self.attention(hidden,enc_output) 
        # context -> bs,units
        # atten_w -> bs,seq_len,1
        x = tf.concat([context,x],axis=-1)
        # x -> bs,emb_dim + units
        output,hidden,state = self.lstm(x,initial_state=[hidden,cell_state])
        # output -> bs,1,units
        # state -> bs,units
        return tf.squeeze(output), hidden, state, atten_w



class LSTM_Resnet(tf.keras.Model):
    def __init__(self,seq_len,emb_dim=4,atten_units=50,final_rep=1):
        """ Arguments
            emb_dim -> embedding dimension at decoder
            seq_len -> sequence length
            atten_units -> hidden dims for Attention
            final_rep -> final output shape

            Returns
            final_op -> final ouput (bs,output_seq,final_rep)
            all_atten -> attention (bs,output_seq,seq_len,1)  

        """
        super().__init__()
        ip_dims = CustomResNet().compute_output_shape((None,None,None,5))[-1]
        self.cnn = layers.TimeDistributed(CustomResNet())
        self.encoder = Encoder(ip_dims)
        self.decoder = Decoder(ip_dims,emb_dim,seq_len,atten_units)
        self.seq_len = seq_len
        self.fc = tf.keras.layers.Dense(final_rep)
        self.decoder_ouput_size = 4
    def call(self,x,training=False):
        bs = x.shape[0]
        decoder_input = tf.convert_to_tensor(np.tile(np.arange(self.decoder_ouput_size),(bs,1)))
        x = self.cnn(x,training=training)
        enc_output,enc_hidden,enc_cell_state,_,_ = self.encoder(x)
        # enc_output -> bs,seq_len,units
        # enc_hidden -> bs,units
        dec_hidden, dec_cell_state = enc_hidden, enc_cell_state
        final_op,all_atten = [],[]
        for i in range(self.decoder_ouput_size):
            dec_output, dec_hidden, dec_cell_state, atten_w =  \
            self.decoder(decoder_input[:,i],dec_hidden,dec_cell_state,enc_output)
            # dec_output = bs,units
            # dec_hiddem = bs,units
            # atten_w = bs,seq_len,1
            final_op.append(dec_output)
            all_atten.append(atten_w)
        final_op = tf.transpose(tf.convert_to_tensor(final_op),perm=(1,0,2))
        final_op = self.fc(final_op)
        all_atten = tf.transpose(tf.convert_to_tensor(all_atten),perm=(1,0,2,3))
        return tf.squeeze(final_op)