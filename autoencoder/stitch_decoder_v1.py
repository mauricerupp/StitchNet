from utilities import *
from l1_loss import custom_loss
from autoencoder_v5 import *
from psnr_stitched import stitched_PSNR


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils import multi_gpu_model


class StitchDecoder(object):

    def __init__(self, input_size, weights_path, normalizer, isTraining):

        encoder_inputs = Input(shape=input_size)

        autoenc = ConvAutoencoder([input_size[0], input_size[1], 3], norm=normalizer, isTraining=False)
        autoenc.load_encoder_weights(weights_path)
        enc = autoenc.encoder

        # encode each image individually through the pre-trained encoder
        encoded_img_list = []
        index = 1
        for i in range(0, input_size[2], 3):
            x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(encoder_inputs)
            encoded_img_list.append(enc(x))
            index += 1

        # concatenate the images and decode them to a final image
        x = Concatenate(axis=3, name='conc_img_features')(encoded_img_list)

        for i in range(5):
            x = dec_block(x, int(1024 / 2 ** i), i + 1, normalizer, isTraining, "D2")
        out = Conv2D(3, 3, activation='tanh', padding='same', name='final_conv')(x)

        self.stitchdecoder = Model(inputs=encoder_inputs, outputs=out, name='stitcher')
        self.stitchdecoder.summary()
        # enable multi-gpu-processing
        self.stitchdecoder = multi_gpu_model(self.stitchdecoder, gpus=2)
        self.stitchdecoder.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy', stitched_PSNR])

    def load_weights(self, path):
        self.stitchdecoder.load_weights(filepath=path)


mod = StitchDecoder(input_size=(64, 64, 15),
                    weights_path='/data/cvg/maurice/logs/ConvAutoencoder_V5fixed_instanceBIGGER_20_80_run3/encoder_logs/',
                    normalizer='instance',
                    isTraining=True)

