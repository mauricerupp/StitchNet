from tensorflow.python.keras.applications.vgg16 import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model
from utilities import *
from tensorflow.python.keras.losses import *
from tensorflow import Graph, Session


def vgg_loss(y_true, y_pred):
    """
    does a weighted loss of perceptual loss and MAE
    :param y_true:
    :param y_pred:
    :return:
    """
    percentage_MAE = 0.2
    percentage_perceptual = 1 - percentage_MAE
    SCALE = 1.68e-5

    # MAE has shape of (64x64), Perceptual has shape of (8x8), therefore we take the mean of those values and
    # add a scale factor, so they have the same general scale, so they can be weighted properly
    return percentage_MAE * K.mean(mean_absolute_error(y_true, y_pred)) + SCALE * percentage_perceptual * perceptual_loss(y_true, y_pred)


def perceptual_loss(y_true, y_pred):
    """
    calculates the perceptual loss for the 4rd block of VGG16, which has a shape of 8x8
    :param y_true:
    :param y_pred:
    :return:
    """

    my_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=[64, 64, 3])
    my_vgg16.trainable = False
    for l in my_vgg16.layers:
        l.trainable = False
    model = Model(inputs=my_vgg16.input, outputs=my_vgg16.get_layer('block4_conv3').output)
    model.trainable = False
    # preprocess input works with data in the range of [0,255], so the images have to be reverted
    #yt_new = tf.keras.applications.vgg16.preprocess_input(revert_zero_center(y_true)*255.0)
    #yp_new = tf.keras.applications.vgg16.preprocess_input(revert_zero_center(y_pred)*255.0)
    # since we here have 8x8=64 pixels, we have to scale the result
    return K.mean(mean_squared_error(model(y_true), model(y_pred)))
