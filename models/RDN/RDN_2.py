import l1_loss
from utilities import *


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def create_model(pretrained_weights=None, input_size=None, G0=64, G=32, D=20, C=6):
    """
    RDN implemented from paper 'Residual Dense Network for Image Super-Resolution
    with a custom conv layer which shares weights over all stacked input images
    and a depth-to-space layer at the end of the pipeline
    The end of the Network is more similar to the actual paper and the best performing one at the moment
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)

    # extract features for every input image
    conv1 = feature_extract(inputs, G0)

    global_conv1 = Conv2D(G0, kernel_size=3, activation='relu', padding='same', name='global_conv1')(conv1)
    global_conv2 = Conv2D(G0, kernel_size=3, activation='relu', padding='same', name='global_conv2')(global_conv1)

    # first RDB
    RDB = create_RDB(global_conv2, 'RDB1', G0, G, C)
    RDBlocks_list = [RDB, ]

    # add the remaining RDB
    for i in range(2, D + 1):
        RDB = create_RDB(RDB, 'RDB' + str(i), G0, G, C)
        RDBlocks_list.append(RDB)

    RDB_out = Concatenate(axis=3)(RDBlocks_list)
    RDB_out = Conv2D(G0, kernel_size=1, padding='same', name='global_1x1_conv')(RDB_out)
    RDB_out = Conv2D(G0, kernel_size=3, padding='same', name='global_conv3')(RDB_out)
    out = Add()([RDB_out, global_conv1])

    # concatenate the very first extracted features with the output of the residual learning
    out = Concatenate(axis=3)([out, conv1, inputs])

    # Upscaling / depth to space
    out = Conv2D(G0, kernel_size=3, padding='same', name='upscale_conv_1')(out)
    out = Conv2D(int(G0 / 2), kernel_size=3, padding='same', name='upscale_conv_2')(out)
    out = Conv2D(G0 * 2, kernel_size=3, padding='same', name='upscale_conv_3')(out)
    out = depth_to_space(out, 2)

    out = Conv2D(int(G0 / 2), kernel_size=3, padding='same')(out)
    out = Conv2D(int(G0 / 4), kernel_size=3, padding='same')(out)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=3, padding='same')(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])

    model.summary()

    # Save the configurations as txt-file
    #with open('RDN ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))

    #plot_model(model, to_file='RDN_2_D{}C{}.png'.format(D, C))

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

#mod = create_model(input_size=(64,64,15), D=3, C=3)