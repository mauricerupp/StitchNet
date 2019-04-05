import loss
import numpy as np

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf


def try_net(pretrained_weights=None, input_size=None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(inputs)
    up1 = UpSampling2D()(conv1)
    up2 = UpSampling2D()(up1)
    conv2 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(up2)

    model = Model(inputs=inputs, outputs=conv2)
    model.compile(optimizer='adam', loss=loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


mod = try_net(input_size=(100,150,9))