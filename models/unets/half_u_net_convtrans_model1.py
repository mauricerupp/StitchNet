import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *


def create_model(pretrained_weights=None, input_size=None):
    """
    This arcitecture was inspired by the approach of progressive upsampling of the image
    with increased depth of the receptive field opposed to standard u-nets:
    each first layer has a filter of 5x5 instead of 3x3 and the second has a dilation of 3 instead of 1
    also the initial learning rate was set higher than the initial value (0.001), this is due to the
    depth of the network, which is quite deep
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)

    conv1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3)(conv5)

    trans6 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                             kernel_initializer='he_normal', strides=(2,2))(conv5)
    merge6 = concatenate([conv4, trans6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    trans7 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                             kernel_initializer='he_normal', strides=(2, 2))(conv6)
    merge7 = concatenate([conv3, trans7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    trans8 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                             kernel_initializer='he_normal', strides=(2, 2))(conv7)
    merge8 = concatenate([conv2, trans8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    trans9 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                             kernel_initializer='he_normal', strides=(2, 2))(conv8)
    merge9 = concatenate([conv1, trans9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    trans10 = Conv2DTranspose(32, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal', strides=(2,2))(conv9)
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(trans10)
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(conv10)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(lr=0.01), loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model