import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import tensorflow as tf


def create_model(pretrained_weights=None, input_size=None):
    """
    idea: process the input in packages of 3 channels at first in order to grasp the features for every picture single
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)
    # get features of every image individually and concatenate them
    # all those layers share the same weights
    conv1_x = Conv2D(64, 3, activation='relu', padding='same', name='conv1_x')
    layerstack = []

    for i in range(0, input_size[2], 3):
        test = conv1_x(inputs[:, :, :, i:i + 3])
        layerstack.append(test)

    conc1 = concatenate([layerstack[i] for i in range(len(layerstack))], axis=3)

    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(conc1)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# ------- Functions -------- #
def feature_extractor(input):
    input_shape = input.get_shape().as_list()
    layerstack = []
    var = 0
    for i in range(0, input_shape[3], 3):
        conv1_x = Conv2D(64, 3, activation='relu', padding='same', name='conv1_{}'.format(var))(input[:, :, :, i:i+3])
        layerstack.append(conv1_x)
        var += 1
    return concatenate([layerstack[i] for i in range(len(layerstack))], axis=3)

# ------- END -------- #


mod = create_model(input_size=(64,64,15))