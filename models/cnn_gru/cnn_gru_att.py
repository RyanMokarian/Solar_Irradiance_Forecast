
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations

from models.cnn_gru.cnn import CNN, Encoder

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
    
class CnnGruAtt(tf.keras.Model):
    def __init__(self, seq_len, emb_dim=4, atten_units=25, final_rep=1):
        """CNN+GRU with attention.
        
        Arguments:
            seq_len {int} -- The length of the image sequence.
            emb_dim {int} -- Embedding dimension of the decoder.
            emb_dim {int} -- Embedding dimension of the decoder.
            atten_units {int} -- Hidden dims for attention.
            final_rep {int} -- Final output shape.

        Returns:
            final_op -- final ouput (bs,output_seq,final_rep)
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
        # Reshape the decoder input for the last batch
        bs = x.shape[0]
        decoder_input = tf.convert_to_tensor(np.tile(np.arange(self.decoder_output_size),(bs,1)))

        # enc_output -> bs, seq_len, units
        # enc_hidden -> bs, units
        encoder_output, encoder_hidden = self.encoder(x)
        
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

        return tf.squeeze(final_op)