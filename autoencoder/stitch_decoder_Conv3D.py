from utilities import *
from l1_loss import custom_loss
from autoencoder_v6 import *
from autoencoder_BIG import *
from psnr_stitched import stitched_PSNR
from ssim_stitched import stitched_ssim
from perceptual_stitched_loss import vgg_loss, mae_loss, perceptual_stitched_loss


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import time
import datetime


class StitchDecoder(object):

    def __init__(self, input_size, weights_path_small=None, weights_path_big=None, normalizer=None, isTraining=None):

        encoder_inputs = Input(shape=input_size)

        # Initialize the small autoencoder
        autoenc_small = ConvAutoencoder([input_size[0], input_size[1], 3], norm="instance", isTraining=False)
        autoenc_small.autoencoder.load_weights(weights_path_small)
        # freeze the weights
        autoenc_small.isNotTraining()

        # Initialize the big autoencoder
        autoenc_big = ConvAutoencoderBIG([2*input_size[0], 2*input_size[1], 3], norm="instance", isTraining=False)
        autoenc_big.autoencoder.load_weights(weights_path_big)
        autoenc_big.isNotTraining()

        # create the encoder E1
        encoder_model = Model(inputs=autoenc_small.autoencoder.input, outputs=autoenc_small.autoencoder.get_layer('autoencoder').
                              get_layer(name='bottleneck_relu_layer').output)
        encoder_model.trainable = False

        # encode each image individually through the pre-trained encoder and apply the same transposed 2D-Conv
        enc_conv = Conv2DTranspose(512, kernel_size=3, padding='same', activation='relu', name='enc_bottleneck_Conv2d', strides=2)
        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(encoder_inputs)
            x = enc_conv(encoder_model(x))
            x = normalize(x, '{}_decoder_norm{}_{}'.format("upsample", str(index), i), 'instance', True)
            x = LeakyReLU()(x)
            x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
            encoded_img_list.append(x)
            index += 1

        # Reshape the list into the form of (batchsize, 5, width, height, channels), (b,5,8,8,512)
        x = Concatenate(axis=-1, name='conc_img_features')(encoded_img_list)
        x = Lambda(lambda x: K.permute_dimensions(x, (0, 4, 1, 2, 3)))(x)

        # apply 3D convolutions with decreasing size of the "samples" dimension
        for i in range(3):
            x = Conv3D(512, 2, strides=(2,1,1), padding='same', activation=None)(x)
            x = normalize(x, 'decoder_norm_{}'.format(i), 'instance', True)
            x = LeakyReLU()(x)

        # get rid of the samples axis
        x = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=1))(x)
        """
        # pass this through the decoder part of the big Autoencoder
        for i in range(37):
            x = autoenc_big.autoencoder.get_layer('autoencoderBIG').get_layer(index=i+42)(x)
        """
        out = x

        self.stitchdecoder = Model(inputs=encoder_inputs, outputs=out, name='stitcher')
        self.stitchdecoder.summary()

        # enable multi-gpu-processing
        self.stitchdecoder = multi_gpu_model(self.stitchdecoder, gpus=2)

        self.stitchdecoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                              loss=vgg_loss, metrics=['accuracy', stitched_PSNR, stitched_ssim,
                                                           perceptual_stitched_loss, mae_loss])

"""
mod = StitchDecoder(input_size=(64, 64, 15),
                    weights_path='/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback/weight_logs/auto_weights-improvement-01.h5',
                    normalizer='instance',
                    isTraining=True)

mod = StitchDecoder(input_size=(64, 64, 15),
                    normalizer='instance',
                    isTraining=True)
"""