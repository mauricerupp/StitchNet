import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def create_model(pretrained_weights=None, input_size=None):
    """
    like model2, but with a higher receptive field and some dilation added at the first layers
    receptive field from start to highest filter level: 101/128
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 13, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 13, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 11, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 11, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 11, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 11, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 7, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 7, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Upsampling
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    out_trans = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=2)(conv9)

    # Deblur
    deblur_CNN_layer1 = Conv2D(filters=128, kernel_size=10, strides=1, padding='same')(out_trans)
    deblur_CNN_layer1 = BatchNormalization()(deblur_CNN_layer1)
    deblur_CNN_layer1 = Activation('relu')(deblur_CNN_layer1)

    deblur_CNN_layer2 = Conv2D(filters=320, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer1)
    deblur_CNN_layer2 = BatchNormalization()(deblur_CNN_layer2)
    deblur_CNN_layer2 = Activation('relu')(deblur_CNN_layer2)

    deblur_CNN_layer3 = Conv2D(filters=320, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer2)
    deblur_CNN_layer3 = BatchNormalization()(deblur_CNN_layer3)
    deblur_CNN_layer3 = Activation('relu')(deblur_CNN_layer3)

    deblur_CNN_layer4 = Conv2D(filters=320, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer3)
    deblur_CNN_layer4 = BatchNormalization()(deblur_CNN_layer4)
    deblur_CNN_layer4 = Activation('relu')(deblur_CNN_layer4)

    deblur_CNN_layer5 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer4)
    deblur_CNN_layer5 = BatchNormalization()(deblur_CNN_layer5)
    deblur_CNN_layer5 = Activation('relu')(deblur_CNN_layer5)

    deblur_CNN_layer6 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(deblur_CNN_layer5)
    deblur_CNN_layer6 = BatchNormalization()(deblur_CNN_layer6)
    deblur_CNN_layer6 = Activation('relu')(deblur_CNN_layer6)

    deblur_CNN_layer7 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer6)
    deblur_CNN_layer7 = BatchNormalization()(deblur_CNN_layer7)
    deblur_CNN_layer7 = Activation('relu')(deblur_CNN_layer7)

    deblur_CNN_layer8 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(deblur_CNN_layer7)
    deblur_CNN_layer8 = BatchNormalization()(deblur_CNN_layer8)
    deblur_CNN_layer8 = Activation('relu')(deblur_CNN_layer8)

    deblur_CNN_layer9 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(deblur_CNN_layer8)
    deblur_CNN_layer9 = BatchNormalization()(deblur_CNN_layer9)
    deblur_CNN_layer9 = Activation('relu')(deblur_CNN_layer9)

    deblur_CNN_layer10 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(deblur_CNN_layer9)
    deblur_CNN_layer10 = BatchNormalization()(deblur_CNN_layer10)
    deblur_CNN_layer10 = Activation('relu')(deblur_CNN_layer10)

    deblur_CNN_layer11 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(deblur_CNN_layer10)
    deblur_CNN_layer11 = BatchNormalization()(deblur_CNN_layer11)
    deblur_CNN_layer11 = Activation('relu')(deblur_CNN_layer11)

    deblur_CNN_layer12 = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(deblur_CNN_layer11)
    deblur_CNN_layer12 = BatchNormalization()(deblur_CNN_layer12)
    deblur_CNN_layer12 = Activation('relu')(deblur_CNN_layer12)

    deblur_CNN_layer13 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(deblur_CNN_layer12)
    deblur_CNN_layer13 = BatchNormalization()(deblur_CNN_layer13)
    deblur_CNN_layer13 = Activation('relu')(deblur_CNN_layer13)

    deblur_CNN_layer14 = Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(deblur_CNN_layer13)
    deblur_CNN_layer14 = BatchNormalization()(deblur_CNN_layer14)
    deblur_CNN_layer14 = Activation('relu')(deblur_CNN_layer14)

    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(deblur_CNN_layer14)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    #model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model