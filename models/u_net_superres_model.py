import loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def create_u_net_superres_model(pretrained_weights=None, input_size=None):

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

mod = create_u_net_superres_model(input_size=(100,150,24))