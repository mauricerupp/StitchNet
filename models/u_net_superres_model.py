from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def create_u_net_superres_model(input_shape, loss, optimizer):
    model = keras.Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='Same', activation='relu',
                     input_shape=input_shape, data_format="channels_last"))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], )
    return model

#model =create_vgg16_model((214, 256, 3), 'mean_squared_error', 'adam')
#model.summary()