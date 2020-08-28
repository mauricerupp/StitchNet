from utilities import *
from psnr_stitched import stitched_PSNR
from ssim_stitched import stitched_ssim
from perceptual_stitched_loss import vgg_loss, mae_loss, perceptual_stitched_loss
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import multi_gpu_model


class StitchDecoder(object):

    def __init__(self, input_size, normalizer, isTraining):

        single_input = Input(shape=[64, 64, 3])
        encoder_inputs = Input(shape=input_size)

        # create the encoderblocks:
        x = single_input
        for i in range(4):
            x = enc_block_leaky(x, int(64 * 2**i), i + 1, normalizer, isTraining)

        # bottleneck of the encoder:
        x = Conv2D(512, 3, activation=None, padding='same', strides=1, name='encoder_conv5_1')(x)
        x = normalize(x, 'encoder_norm5_1', normalizer, isTraining)
        x = LeakyReLU()(x)
        x = Conv2D(512, 3, activation=None, padding='same', strides=1, name='encoder_conv5_2')(x)
        x = normalize(x, 'encoder_norm5_2', normalizer, isTraining)
        enc_out = LeakyReLU(name='bottleneck_relu_layer')(x)

        # create the encoder, which is the same for every snapshot
        encoder_model = Model(inputs=single_input, outputs=enc_out)

        # encode each snapshot individually through the same encoder (so weights are shared across the snapshots)
        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(encoder_inputs)
            encoded_img_list.append(encoder_model(x))
            index += 1

        # concatenate the encodings
        x = Concatenate(axis=3, name='conc_img_features')(encoded_img_list)

        # global convolutions
        for i in range(2):
            x = Conv2D(int(2048 / 2 ** i), 3, activation=None, padding='same', strides=1,
                       name='{}_decoder_conv{}_{}'.format("conc_conv", index, i + 2))(x)
            x = normalize(x, '{}_decoder_norm{}_{}'.format("conc_conv", index, i + 2), 'instance', True)
            x = LeakyReLU()(x)

        # decoder blocks
        for i in range(5):
            x = big_dec_block_leaky(x, int(1024 / 2 ** i), i + 1, normalizer, isTraining, "D2")
        out = Conv2D(3, 3, activation='tanh', padding='same', name='final_conv')(x)

        self.stitchdecoder = Model(inputs=encoder_inputs, outputs=out, name='stitcher')
        self.stitchdecoder.summary()

        # enable multi-gpu-processing
        encoder_model = multi_gpu_model(encoder_model, gpus=2)
        self.stitchdecoder = multi_gpu_model(self.stitchdecoder, gpus=2)

        self.stitchdecoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                                   loss=vgg_loss, metrics=['accuracy', stitched_PSNR, stitched_ssim,
                                                           perceptual_stitched_loss, mae_loss])