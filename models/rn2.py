import l1_loss
from utilities import *
from psnr_stitched import stitched_PSNR
from ssim_stitched import stitched_ssim
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import tensorflow as tf


def create_model(pretrained_weights=None, input_size=None, filter_size=128, block_amount=12, normalizer=None):
    """
    A simple residual network with only ResBlocks
    """
    inputs = Input(input_size)

    # first global conv, which has no normalization
    global_conv1 = Conv2D(filter_size*2, kernel_size=7, activation='relu', padding='same', name='global_conv1')(inputs)

    # second global conv with normalization
    global_conv2 = Conv2D(filter_size, kernel_size=7, padding='same', name='global_conv2')(global_conv1)
    global_conv2 = normalize(name="globalconv2_norm1", input_layer=global_conv2, normalizer=normalizer)
    global_conv2 = Activation('relu')(global_conv2)

    # first RB
    RB = create_resblock(prior_layer=global_conv2, block_name='RB1',
                         n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer)

    # add the remaining RBs
    for i in range(2, block_amount + 1):
        RB = create_resblock(prior_layer=RB, block_name='RB' + str(i),
                             n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer)

    # depth to space
    out = depth_to_space(RB, 2)

    out = Conv2D(16, kernel_size=5, padding='same', activation='relu')(out)
    out = Conv2D(8, kernel_size=5, padding='same', activation='relu')(out)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=l1_loss.custom_loss, metrics=['accuracy', stitched_PSNR, stitched_ssim])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model