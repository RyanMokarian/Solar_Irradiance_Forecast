import tensorflow as tf

class DummyModel(tf.keras.Model):
      def __init__(self):
        super(DummyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation=None)

      def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        return self.dense2(x)
    
# TODO : Add models

    
