from utilities import *
from psnr import PSNR


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import numpy as np
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model


class ConvAutoencoder(object):

    def __init__(self, input_size, norm='batch', isTraining=None):

        inputs = Input(shape=input_size, name='encoder_input')
        x = inputs

        # encoder (add blocks until 4x4x256):
        for i in range(4):
            x = enc_block(x, int(64 * 2**i), i + 1, norm, isTraining)

        # bottom of the net (like U-Net):
        x = Conv2D(256, 3, activation=None, padding='same', strides=1, name='encoder_conv5_1')(x)
        x = normalize(x, 'encoder_norm5_1', norm, isTraining)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, activation=None, padding='same', strides=1, name='encoder_conv5_2')(x)
        x = normalize(x, 'encoder_norm5_2', norm, isTraining)
        enc_out = Activation('relu')(x)

        # decoder
        x = enc_out
        for i in range(4):
            x = dec_block(x, int(128 / 2**i), i + 1, norm, isTraining, "autoenc_v4")
        # final layer
        out = Conv2D(3, 3, activation='tanh', padding='same', strides=1, name='final_conv')(x)

        self.encoder = Model(inputs, enc_out, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=out, name='autoencoder')

        #self.autoencoder.summary()
        # self.autoencoder = multi_gpu_model(self.autoencoder, gpus=2)

        if not isTraining:
            self.encoder.trainable = False
            self.autoencoder.trainable = False

        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy', PSNR])

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)

    def load_weights(self, path):
        self.autoencoder.load_weights(filepath=path)

#mod = ConvAutoencoder(input_size=(64,64,3), norm='batch', isTraining=True)