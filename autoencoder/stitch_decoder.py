from utilities import *
from l1_loss import custom_loss
from autoencoder_v2 import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K
import numpy as np


class StitchDecoder(object):

    def __init__(self, input_size, weights_path):

        encoder_inputs = Input(shape=input_size)

        autoenc = ConvAutoencoder([input_size[0], input_size[1], 3])
        autoenc.load_weights(weights_path)
        enc = autoenc.encoder

        # encode each image individually through the pre-trained encoder
        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(encoder_inputs)

            encoded_img_list.append(enc(x))
            index += 1

        # concatenate the images and decode them to a final image
        conc = Concatenate(axis=3, name='conc_img_features')(encoded_img_list)
        conv = Conv2DTranspose(640, 3, activation='relu', padding='same', name='decoder_conv1', strides=2)(conc)
        conv = Conv2DTranspose(320, 3, activation='relu', padding='same', name='decoder_conv2', strides=2)(conv)
        conv = Conv2DTranspose(160, 3, activation='relu', padding='same', name='decoder_conv3', strides=2)(conv)
        conv = Conv2DTranspose(80, 3, activation='relu', padding='same', name='decoder_conv4', strides=2)(conv)
        conv = Conv2DTranspose(40, 3, activation='relu', padding='same', name='decoder_conv5', strides=2)(conv)
        out = Conv2D(3, 3, activation='tanh', padding='same', name='final_conv')(conv)

        self.stitchdecoder = Model(inputs=encoder_inputs, outputs=out, name='stitcher')
        self.stitchdecoder.summary()
        self.stitchdecoder.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

    def load_weights(self, path):
        self.stitchdecoder.load_weights(filepath=path)


#mod = StitchDecoder(input_size=(64, 64, 15), encoder_weights_path='/home/maurice/Dokumente/encoder_logs/')