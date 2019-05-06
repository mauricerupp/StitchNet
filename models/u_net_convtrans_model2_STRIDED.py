import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


def create_model(pretrained_weights=None, input_size=None):
    """
    same receptive field as model2, but used strides instead of maxpooling (idea from Perceptual Losses for Real-Time style...)

    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(inputs)
    conv1_2 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=2)(conv1)
    conv1_2 = BatchNormalization()(conv1_2)

    conv2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv1_2)
    conv2_2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=2)(conv2)
    conv2_2 = BatchNormalization()(conv2_2)

    conv3 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv2_2)
    conv3_2 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=2)(conv3)
    conv3_2 = BatchNormalization()(conv3_2)

    conv4 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv3_2)
    conv4_2 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal', strides=2)(conv4)

    conv5 = Conv2D(1024, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_2)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2DTranspose(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv5)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv6)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    convtrans1 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2,2))(conv9)

    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(convtrans1)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model