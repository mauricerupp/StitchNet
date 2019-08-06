from utilities import *
from psnr import PSNR
from perceptual_loss import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import numpy as np
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model
import datetime
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.applications.vgg19 import VGG19


class ConvAutoencoder(object):

    def __init__(self, input_size, norm='batch', isTraining=None):
        """
        Same as V5, but lower lr and leaky_relu instead of relu
        Also random crops, without downsampling big images
        :param input_size:
        :param norm:
        :param isTraining:
        """
        inputs = Input(shape=input_size, name='encoder_input')
        x = inputs

        # encoder (add blocks until 4x4x256):
        for i in range(4):
            x = enc_block_leaky(x, int(64 * 2**i), i + 1, norm, isTraining)

        # bottom of the net (like U-Net):
        x = Conv2D(512, 3, activation=None, padding='same', strides=1, name='encoder_conv5_1')(x)
        x = normalize(x, 'encoder_norm5_1', norm, isTraining)
        x = LeakyReLU()(x)
        x = Conv2D(512, 3, activation=None, padding='same', strides=1, name='encoder_conv5_2')(x)
        x = normalize(x, 'encoder_norm5_2', norm, isTraining)
        enc_out = LeakyReLU()(x)

        # decoder
        x = enc_out
        for i in range(4):
            x = dec_block_leaky(x, int(256 / 2**i), i + 1, norm, isTraining, "autoenc_v6")
        # final layer
        out = Conv2D(3, 3, activation='tanh', padding='same', strides=1, name='final_conv')(x)

        self.encoder = Model(inputs, enc_out, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=out, name='autoencoder')

        if not isTraining:
            self.encoder.trainable = False
            for l in self.encoder.layers:
                l.trainable = False
            self.autoencoder.trainable = False
            for l in self.autoencoder.layers:
                l.trainable = False

        #self.autoencoder.summary()
        #self.encoder.summary()
        self.autoencoder = multi_gpu_model(self.autoencoder, gpus=2)
        self.encoder = multi_gpu_model(self.encoder, gpus=2)

        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                                 loss=vgg_loss,
                                 metrics=['accuracy', PSNR])

        #with open('Autoenc_v4 ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
        #    self.autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

        #plot_model(self.autoencoder, to_file='Autoencoder_v4')

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)

    def load_weights(self, path):
        self.autoencoder.load_weights(filepath=path)


#mod = ConvAutoencoder(input_size=(64,64,3), norm='instance', isTraining=True)