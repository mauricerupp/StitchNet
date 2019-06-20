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

    def __init__(self, input_size, norm='batch', training_flag=None):

        inputs = Input(shape=input_size, name='encoder_input')
        x = inputs

        # encoder (add blocks until 4x4x256):
        for i in range(4):
            x = enc_block(x, int(64 * 2**i), i+1, norm, training_flag)

        # bottom of the net (like U-Net):
        x = Conv2D(256, 3, activation=None, padding='same', strides=1, name='encoder_conv5_1')(x)
        x = normalize(x, 'encoder_norm5_1', norm, training_flag)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, activation=None, padding='same', strides=1, name='encoder_conv5_2')(x)
        x = normalize(x, 'encoder_norm5_2', norm, training_flag)
        enc_out = Activation('relu')(x)

        # decoder
        x = enc_out
        for i in range(4):
            x = dec_block(x, int(128 / 2**i), i+1, norm, training_flag)
        # final layer
        out = Conv2D(3, 3, activation='tanh', padding='same', strides=1, name='final_conv')(x)

        self.encoder = Model(inputs, enc_out, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=out, name='autoencoder')

        #self.autoencoder.summary()
        # self.autoencoder = multi_gpu_model(self.autoencoder, gpus=2)

        if not training_flag:
            self.encoder.trainable = False
            self.autoencoder.trainable = False

        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy', PSNR])

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)

    def load_weights(self, path):
        self.autoencoder.load_weights(filepath=path)


def enc_block(input_layer, filters, index, normalizer, training_flag):
    if filters > 256:
        filters = 256
    if index == 1:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = Activation('relu')(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = normalize(x, 'encoder_norm{}_1'.format(index), normalizer, training_flag)
        x = Activation('relu')(x)

    x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_2'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_2'.format(index), normalizer, training_flag)
    x = Activation('relu')(x)
    if filters < 256:
        x = Conv2D(filters*2, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_3'.format(index), normalizer, training_flag)
    return Activation('relu')(x)


def dec_block(input_layer, filters, index, normalizer, training_flag):
    x = Conv2DTranspose(filters, 3, activation=None, padding='same', name='decoder_conv{}_1'.format(index), strides=2)(input_layer)
    x = normalize(x, 'decoder_norm{}_1'.format(index), normalizer, training_flag)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='decoder_conv{}_2'.format(index))(x)
    x = normalize(x, 'decoder_norm{}_2'.format(index), normalizer, training_flag)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='decoder_conv{}_3'.format(index))(x)
    x = normalize(x, 'decoder_norm{}_3'.format(index), normalizer, training_flag)
    return Activation('relu')(x)


#mod = ConvAutoencoder(input_size=(64,64,3), training_flag=True)