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
        out1 = Conv2D(3, 3, activation='tanh', padding='same', strides=1, name='final_conv_2')(x)
        out2 = Conv2D(3, 3, activation='tanh', padding='same', strides=1, name='final_conv_1')(x)

        self.encoder = Model(inputs, enc_out, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=[out1, out2],
                                 name='autoencoder')

        #self.autoencoder.summary()
        self.autoencoder = multi_gpu_model(self.autoencoder, gpus=2)
        self.encoder = multi_gpu_model(self.autoencoder, gpus=2)

        if not isTraining:
            self.encoder.trainable = False
            for l in self.encoder.layers:
                l.trainable = False
            self.autoencoder.trainable = False
            for l in self.autoencoder.layers:
                l.trainable = False

        self.autoencoder.compile(optimizer='adam',
                                 loss=[vgg_loss, 'mean_absolute_error'],
                                 loss_weights=[0.5, 0.5],
                                 metrics=['accuracy', PSNR])

        #with open('Autoenc_v4 ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
        #    self.autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

        #plot_model(self.autoencoder, to_file='Autoencoder_v4')

    def load_encoder_weights(self, path):
        self.encoder.load_weights(filepath=path)

    def load_weights(self, path):
        self.autoencoder.load_weights(filepath=path)


#mod = ConvAutoencoder(input_size=(64,64,3), norm='instance', isTraining=True)