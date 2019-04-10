import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.python.keras.utils import plot_model


def create_u_net_superres_model(pretrained_weights=None, input_size=None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = ZeroPadding2D(((1, 0), (1, 0)))(up7)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = ZeroPadding2D(((0, 0), (1, 0)))(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # super-res-net:
    #1. zero padding 2. conv 64 3, 3. relu, inputsize = outputsize!
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    conv13 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    conv14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
    conv15 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    conv16 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)
    conv17 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv16)
    conv18 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    conv19 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)
    conv20 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv19)
    conv21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv20)
    conv22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv21)
    conv23 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv22)
    conv24 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv23)
    conv25 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv24)
    conv26 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv25)
    conv27 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv26)
    conv28 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv27)
    conv29 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv28)
    conv30 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv29)
    conv31 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv30)
    up10 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv31))
    up11 = UpSampling2D(size=(2, 2))(up10)
    out = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=1)(up11)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model