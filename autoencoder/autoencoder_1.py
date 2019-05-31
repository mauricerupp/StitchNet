from utilities import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import tensorflow.keras.backend as K
import datetime


class ConvAutoencoder(object):
    def __init__(self, input_size):
        # setup to encoding/decoding
        inputs = Input(input_size, name='encoder_input')
        encoder_layers = encode(inputs)

        decoder_inputs = Input(K.int_shape(encoder_layers)[1:], name='decoder_input')
        decoder_layers = decode(decoder_inputs)

        self.encoder = Model(inputs, encoder_layers, name='encoder')
        self.decoder = Model(decoder_inputs, decoder_layers, name='decoder')

        self.autoencoder = Model(inputs, self.decoder(self.encoder.output), name='autoencoder')
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)


#mod = ConvAutoencoder(input_size=(64,64,3))