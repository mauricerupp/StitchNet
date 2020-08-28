import l1_loss
from psnr_stitched import stitched_PSNR
from ssim_stitched import stitched_ssim
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import multi_gpu_model


def create_model(pretrained_weights=None, input_size=None):
    """
    A fairly standard U-Net with skip connections and an increased receptive field
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 15, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 15, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 13, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 13, activation='relu', padding='same', dilation_rate=2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 13, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 13, activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 9, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 9, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 6, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(512, 2, activation='relu', padding='same', strides=(2, 2))(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, activation='relu', padding='same', strides=(2, 2))(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, activation='relu', padding='same', strides=(2, 2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, activation='relu', padding='same', strides=(2, 2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    up10 = Conv2DTranspose(32, 2, activation='relu', padding='same', strides=(2, 2))(conv9)
    conv10 = Conv2D(32, 3, activation='relu', padding='same')(up10)
    out = Conv2D(3, 3, activation='tanh', padding='same', strides=1)(conv10)
    model = Model(inputs=inputs, outputs=out)
    model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy', stitched_PSNR, stitched_ssim])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model