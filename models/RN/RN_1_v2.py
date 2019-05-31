import l1_loss
from utilities import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def create_model(pretrained_weights=None, input_size=None, filter_size=128, block_amount=12, normalizer=None):
    """
    A simple residual network and a Convolution layer for every input image with shared weights
    :param pretrained_weights:
    :param input_size:
    :param filter_size:
    :param block_amount:
    :param normalizer:
    :return:
    """
    inputs = Input(input_size)
    # feature extractor of all images in the same convolution
    conv1 = feature_extract(inputs, filter=64, kernel=5)

    global_conv2 = Conv2D(filter_size, kernel_size=3, padding='same', name='global_conv')(conv1)
    global_conv2 = normalize(name="globalconv2_norm1", input_layer=global_conv2, normalizer=normalizer)
    global_conv2 = Activation('relu')(global_conv2)

    # first RB
    RB = create_resblock(prior_layer=global_conv2, block_name='RB1',
                         n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer)

    # add the remaining RB
    for i in range(2, block_amount + 1):
        RB = create_resblock(prior_layer=RB, block_name='RB' + str(i),
                             n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer)

    # depth to space
    out = depth_to_space(RB, 2)

    # TODO: Add layers here?
    out = Conv2D(64, kernel_size=5, padding='same', activation='relu')(out)
    out = Conv2D(32, kernel_size=5, padding='same', activation='relu')(out)
    out = Conv2D(16, kernel_size=5, padding='same', activation='relu')(out)
    out = Conv2D(8, kernel_size=5, padding='same', activation='relu')(out)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    # Save the configurations as txt-file
    #with open('RDN ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))

    #plot_model(model, to_file='RN_1.png')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


#mod = create_model(input_size=(64,64,15), filter_size=320, block_amount=8, normalizer='batch')