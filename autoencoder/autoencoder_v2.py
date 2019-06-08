from utilities import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K


class ConvAutoencoder(object):

    def __init__(self, input_size):
        # setup to encoding/decoding
        inputs = Input(shape=input_size, name='encoder_input')

        encoder_layers = encode(inputs, 5)
        decoder_layers = single_decode(encoder_layers, 5)

        self.encoder = Model(inputs, encoder_layers, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=decoder_layers, name='autoencoder')
        #self.autoencoder.summary()
        self.encoder.trainable = False
        self.autoencoder.trainable = False
        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)

    def load_weights(self, path):
        self.autoencoder.load_weights(filepath=path)


#mod = ConvAutoencoder(input_size=(64,64,3))