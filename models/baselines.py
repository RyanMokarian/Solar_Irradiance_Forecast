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

class SunsetModel(tf.keras.Model):
    def __init__(self):
        super(SunsetModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(12, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(24, (3, 3), activation='relu', padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.maxpooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        # Conv block 1
        x = self.conv1(inputs)
        x = self.batch_norm(x)
        x = self.maxpooling(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.maxpooling(x)
        
        # Fully connected
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x