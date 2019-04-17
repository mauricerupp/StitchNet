import l2_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.python.keras.utils import plot_model


def create_model(pretrained_weights=None, input_size=None):
    inputs = Input(input_size)

    #HIDDEN LAYERS
    deblur_CNN_layer1 = Conv2D(filters=128, kernel_size=10, strides=1, padding='same')(input)
    deblur_CNN_layer1 = BatchNormalization()(deblur_CNN_layer1)
    deblur_CNN_layer1 = Activation('relu')(deblur_CNN_layer1)

    deblur_CNN_layer2 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer1)
    deblur_CNN_layer2 = BatchNormalization()(deblur_CNN_layer2)
    deblur_CNN_layer2 = Activation('relu')(deblur_CNN_layer2)

    deblur_CNN_layer3 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer2)
    deblur_CNN_layer3 = BatchNormalization()(deblur_CNN_layer3)
    deblur_CNN_layer3 = Activation('relu')(deblur_CNN_layer3)

    deblur_CNN_layer4 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer3)
    deblur_CNN_layer4 = BatchNormalization()(deblur_CNN_layer4)
    deblur_CNN_layer4 = Activation('relu')(deblur_CNN_layer4)

    deblur_CNN_layer5 = Conv2D(filters=128, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer4)
    deblur_CNN_layer5 = BatchNormalization()(deblur_CNN_layer5)
    deblur_CNN_layer5 = Activation('relu')(deblur_CNN_layer5)

    deblur_CNN_layer6 = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same')(deblur_CNN_layer5)
    deblur_CNN_layer6 = BatchNormalization()(deblur_CNN_layer6)
    deblur_CNN_layer6 = Activation('relu')(deblur_CNN_layer6)

    deblur_CNN_layer7 = Conv2D(filters=512, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer6)
    deblur_CNN_layer7 = BatchNormalization()(deblur_CNN_layer7)
    deblur_CNN_layer7 = Activation('relu')(deblur_CNN_layer7)

    deblur_CNN_layer8 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer7)
    deblur_CNN_layer8 = BatchNormalization()(deblur_CNN_layer8)
    deblur_CNN_layer8 = Activation('relu')(deblur_CNN_layer8)

    deblur_CNN_layer9 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer8)
    deblur_CNN_layer9 = BatchNormalization()(deblur_CNN_layer9)
    deblur_CNN_layer9 = Activation('relu')(deblur_CNN_layer9)

    deblur_CNN_layer10 = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same')(deblur_CNN_layer9)
    deblur_CNN_layer10 = BatchNormalization()(deblur_CNN_layer10)
    deblur_CNN_layer10 = Activation('relu')(deblur_CNN_layer10)

    deblur_CNN_layer11 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer10)
    deblur_CNN_layer11 = BatchNormalization()(deblur_CNN_layer11)
    deblur_CNN_layer11 = Activation('relu')(deblur_CNN_layer11)

    deblur_CNN_layer12 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer11)
    deblur_CNN_layer12 = BatchNormalization()(deblur_CNN_layer12)
    deblur_CNN_layer12 = Activation('relu')(deblur_CNN_layer12)

    deblur_CNN_layer13 = Conv2D(filters=256, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer12)
    deblur_CNN_layer13 = BatchNormalization()(deblur_CNN_layer13)
    deblur_CNN_layer13 = Activation('relu')(deblur_CNN_layer13)

    deblur_CNN_layer14 = Conv2D(filters=64, kernel_size=7, strides = 1, padding='same')(deblur_CNN_layer13)
    deblur_CNN_layer14 = BatchNormalization()(deblur_CNN_layer14)
    deblur_CNN_layer14 = Activation('relu')(deblur_CNN_layer14)

    out = Conv2D(filters=3, kernel_size=7, strides = 1, padding='same', activation='relu')(deblur_CNN_layer14)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l2_loss.my_loss_l2(), metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

mod = create_model(input_size=(128,128,27))