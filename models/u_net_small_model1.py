import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.python.keras.utils import plot_model


def create_model(pretrained_weights=None, input_size=None):
    """
    inserted a u-net similar to the proposed net in the paper, but with an increased receptive field

    :param pretrained_weights:
    :param input_size:
    :return:
    """
    # U-Net:
    inputs = Input(input_size)
    conv1 = Conv2D(64, 11, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 11, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 11, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 11, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 11, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 11, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 9, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 9, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)

    up7 = Conv2DTranspose(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(drop4)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Upsampling (analougous to the u-net structure):
    convtrans1 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv9)
    conv10 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal')(convtrans1)
    conv10 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizers.Adam(lr=0.005), loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
