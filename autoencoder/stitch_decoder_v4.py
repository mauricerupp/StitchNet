from utilities import *
from l1_loss import custom_loss
from autoencoder_v6 import *
from psnr_stitched import stitched_PSNR
from ssim_stitched import stitched_ssim
from perceptual_stitched_loss import vgg_loss, mae_loss, perceptual_stitched_loss


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils import multi_gpu_model


class StitchDecoder(object):

    def __init__(self, input_size, weights_path, normalizer, isTraining):

        encoder_inputs = Input(shape=input_size)

        autoenc = ConvAutoencoder([input_size[0], input_size[1], 3], norm=normalizer, isTraining=False)
        autoenc.autoencoder.load_weights(weights_path)

        # create the encoder
        encoder_model = Model(inputs=autoenc.autoencoder.input, outputs=autoenc.autoencoder.get_layer('autoencoder').
                              get_layer(name='bottleneck_relu_layer').output)

        # encode each image individually through the pre-trained encoder
        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(encoder_inputs)
            encoded_img_list.append(encoder_model(x))
            index += 1

        # concatenate the images and decode them to a final image
        x = Concatenate(axis=3, name='conc_img_features')(encoded_img_list)
        for i in range(2):
            x = Conv2D(int(2048 / 2 ** i), 3, activation=None, padding='same', strides=1,
                       name='{}_decoder_conv{}_{}'.format("conc_conv", index, i + 2))(x)
            x = normalize(x, '{}_decoder_norm{}_{}'.format("conc_conv", index, i + 2), 'instance', True)
            x = LeakyReLU()(x)

        for i in range(5):
            x = big_dec_block_leaky(x, int(1024 / 2 ** i), i + 1, normalizer, isTraining, "D2")
        out = Conv2D(3, 3, activation='tanh', padding='same', name='final_conv')(x)

        self.stitchdecoder = Model(inputs=encoder_inputs, outputs=out, name='stitcher')
        self.stitchdecoder.summary()
        # enable multi-gpu-processing
        self.stitchdecoder = multi_gpu_model(self.stitchdecoder, gpus=2)
        self.stitchdecoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                                   loss=vgg_loss, metrics=['accuracy', stitched_PSNR, stitched_ssim,
                                                           perceptual_stitched_loss, mae_loss])

    def load_weights(self, path):
        self.stitchdecoder.load_weights(filepath=path)

"""
mod = StitchDecoder(input_size=(64, 64, 15),
                    weights_path='/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback/weight_logs/auto_weights-improvement-01.h5',
                    normalizer='instance',
                    isTraining=True)

mod = StitchDecoder(input_size=(64, 64, 15),
                    encoderweights_path='/home/maurice/Dokumente/encoder_logs/',
                    normalizer='instance',
                    isTraining=True)
"""