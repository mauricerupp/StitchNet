from utilities import *
from l1_loss import custom_loss
from autoencoder_v1 import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow.keras.backend as K
import numpy as np



class StitchEncoder(object):

    def __init__(self, input_size, encoder_weights_path):
        # setup to encoding/decoding
        inputs = Input(shape=input_size, name='encoder_input')

        autoenc = ConvAutoencoder([input_size[0], input_size[1], 3])
        autoenc.load_encoder_weights(encoder_weights_path)

        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(inputs)
            x = np.expand_dims(x, axis=0)
            encoded_img_list.append(autoenc.encoder.predict(x))
            index += 1

        conc = Concatenate(axis=3, name='conc_img_features')(encoded_img_list)
        conv = Conv2DTranspose(128, 3, activation='relu', padding='same', name='decoder_conv1', strides=2)(conc)
        conv = Conv2DTranspose(64, 5, activation='relu', padding='same', name='decoder_conv2', strides=2)(conv)
        conv = Conv2DTranspose(32, 5, activation='relu', padding='same', name='decoder_conv3', strides=2)(conv)
        conv = Conv2DTranspose(16, 5, activation='relu', padding='same', name='decoder_conv4', strides=2)(conv)
        conv = Conv2DTranspose(3, 5, activation='tanh', padding='same', name='decoder_conv5', strides=2)(conv)
        out = Conv2D(3, 3, activation='relu', padding='same', name='encoder_conv1')(conv)

        #self.encoder = Model(inputs, encoder_layers, name='encoder')
        self.stitchencoder = Model(inputs=inputs, outputs=out, name='autoencoder')
        self.stitchencoder.summary()
        self.stitchencoder.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

    def load_weights(self, path):
        self.stitchencoder.load_weights(filepath=path)


#mod = StitchEncoder(input_size=(64,64,15))