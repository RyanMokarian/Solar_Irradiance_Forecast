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



class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(18)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_op, values, enc_stamps):
    
        # dec_op is decoder output/hidden state -> (bs,units)
        # values is entire encoder hidden states -> (bs,seq_len,units)
        # enc_stamps ->(bs,38)
        vals = self.W1(values)
        enc_stamps = self.W3(enc_stamps)
        dec_op = self.W2(dec_op)
        dec_op = tf.tile(dec_op[:,None,:],[1,values.shape[1],1])
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(tf.concat([enc_stamps,vals,dec_op],axis=-1)))
        # attention_weights shape == (batch_size, seq_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size,1 hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1, keepdims=True)
        return context_vector, attention_weights



class Decoder(tf.keras.Model):
    def __init__(self,units,seq_len,atten_units):
        # atten_units are intermediate dims for attention. 
        super().__init__()
        self.embedding_month = tf.keras.layers.Embedding(12,6)
        self.embedding_day = tf.keras.layers.Embedding(31,16)
        self.embedding_hour = tf.keras.layers.Embedding(24,12)
        self.embedding_minute = tf.keras.layers.Embedding(4,4)
        self.units = units
        self.lstm = tf.keras.layers.LSTM(self.units,return_sequences=True,return_state=True)
        self.attention = LuongAttention(atten_units)
    
            
    def concat_embedding(self,stamps):
        month = self.embedding_month(stamps[...,0])
        day = self.embedding_day(stamps[...,1])
        hour = self.embedding_hour(stamps[...,2])
        minute = self.embedding_minute(stamps[...,3])
        return tf.concat([month,day,hour,minute],axis=-1)
    
    def call(self,x,hidden,cell_state,enc_output,enc_stamps):
        x = self.concat_embedding(x)[:,None,:]
        enc_stamps = self.concat_embedding(enc_stamps)
        #print(enc_stamps.shape)
        # context -> bs,units
        # atten_w -> bs,seq_len,1
        # x -> bs,emb_dim + units
        _,hidden,state = self.lstm(x,initial_state=[hidden,cell_state])
        
        output,atten_w = self.attention(hidden,enc_output,enc_stamps) 
        # output -> bs,1,units
        # state -> bs,units
        return tf.squeeze(output), hidden, state, atten_w



class LSTM_Resnet(tf.keras.Model):
    def __init__(self,seq_len,atten_units=50,final_rep=1):
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
        ip_dims = CustomResNet().compute_output_shape((None,None,None,5))[-1] + 1
        self.cnn = layers.TimeDistributed(CustomResNet())
        self.encoder = Encoder(ip_dims)
        self.decoder = Decoder(ip_dims,seq_len,atten_units)
        self.seq_len = seq_len
        self.fc = tf.keras.layers.Dense(final_rep)
        self.decoder_ouput_size = 4
    def call(self,x,time_steps,csky,training=False):
        decoder_input = time_steps[:,-4:,:]
        encoder_stamps = time_steps[:,:self.seq_len,:]
        x = self.cnn(x,training=training)
        x = tf.concat([x,csky[...,None]],axis=-1)
        enc_output,enc_hidden,enc_cell_state,_,_ = self.encoder(x)
        # enc_output -> bs,seq_len,units
        # enc_hidden -> bs,units
        dec_hidden, dec_cell_state = enc_hidden, enc_cell_state
        final_op,all_atten = [],[]
        for i in range(self.decoder_ouput_size):
            dec_output, dec_hidden, dec_cell_state, atten_w =  \
            self.decoder(decoder_input[:,i],dec_hidden,dec_cell_state,enc_output,encoder_stamps)
            # dec_output = bs,units
            # dec_hiddem = bs,units
            # atten_w = bs,seq_len,1
            final_op.append(dec_output)
            all_atten.append(atten_w)
        final_op = tf.transpose(tf.convert_to_tensor(final_op),perm=(1,0,2))
        final_op = self.fc(final_op)
        all_atten = tf.transpose(tf.convert_to_tensor(all_atten),perm=(1,0,2,3))
        return tf.squeeze(final_op)
