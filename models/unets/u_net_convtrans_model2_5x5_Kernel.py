import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


def create_model(pretrained_weights=None, input_size=None):
    """
    all 5x5 kernels, receptive field of 109
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(inputs)
    conv1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(pool1)
    conv2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(pool2)
    conv3 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(pool3)
    conv4 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv5)
    drop5 = Dropout(0.2)(conv5)

    up6 = Conv2DTranspose(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(merge6)
    conv6 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv6)

    up7 = Conv2DTranspose(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv6)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(merge7)
    conv7 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv7)

    up8 = Conv2DTranspose(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(merge8)
    conv8 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv8)

    up9 = Conv2DTranspose(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(merge9)
    conv9 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv9)
    convtrans1 = Conv2DTranspose(32, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv9)

    out = Conv2D(3, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(convtrans1)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model