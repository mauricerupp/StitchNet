from utilities import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K


class ConvAutoencoder(object):

    def __init__(self, input_size):
        # setup to encoding/decoding
        inputs = Input(shape=input_size, name='encoder_input')

        conv = Conv2D(64, 3, activation='relu', padding='same', name='encoder_conv1')(inputs)
        conv = Conv2D(128, 3, activation='relu', padding='same', name='encoder_conv2', strides=2)(conv)
        conv = Conv2D(256, 3, activation='relu', padding='same', name='encoder_conv3', strides=2)(conv)
        conv = Conv2D(256, 3, activation='relu', padding='same', name='encoder_conv4', strides=2)(conv)
        conv = Conv2D(256, 3, activation='relu', padding='same', name='encoder_conv5', strides=2)(conv)

        deconv = Conv2DTranspose(128, 3, activation='relu', padding='same', name='decoder_conv1', strides=2)(conv)
        deconv = Conv2DTranspose(64, 3, activation='relu', padding='same', name='decoder_conv2', strides=2)(deconv)
        deconv = Conv2DTranspose(32, 3, activation='relu', padding='same', name='decoder_conv3', strides=2)(deconv)
        deconv = Conv2DTranspose(16, 3, activation='relu', padding='same', name='decoder_conv4', strides=2)(deconv)
        deconv =  Conv2DTranspose(3, 3, activation='tanh', padding='same', name='decoder_conv5', strides=1)(deconv)
        #encoder_layers = encode(inputs)
        #decoder_layers = decode(encoder_layers)

        #self.encoder = Model(inputs, encoder_layers, name='encoder')

        self.autoencoder = Model(inputs=inputs, outputs=deconv, name='autoencoder')
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)


#mod = ConvAutoencoder(input_size=(64,64,3))