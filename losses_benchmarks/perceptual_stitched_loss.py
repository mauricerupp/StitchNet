from tensorflow.python.keras.applications.vgg16 import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model
from utilities import *
from tensorflow.python.keras.losses import *
from l1_loss import custom_loss

percentage_MAE = 0.2
percentage_perceptual = 1 - percentage_MAE
SCALE = 5.57779e-05


def vgg_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: a weighted loss of perceptual loss and MAE of an image with covered and uncovered area
    """
    return mae_loss(y_true, y_pred) + perceptual_stitched_loss(y_true, y_pred)


def mae_loss(y_true, y_pred):
    global percentage_MAE
    return percentage_MAE * custom_loss(y_true, y_pred)


def perceptual_stitched_loss(y_true, y_pred):
    """
    calculates the perceptual loss for the 4rd block of VGG16, which has a shape of 8x8 and is evaluated on
    pictures in [0,255].
    Since this project works with pictures in [-1,1], a empirically evaluated scale factor is introduced.
    :param y_true:
    :param y_pred:
    :return: the weighted perceptual loss of y_true and y_pred
    """
    global SCALE
    global percentage_perceptual
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]

    # create the VGG16 Network and freeze its weights
    my_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=[64, 64, 3])
    my_vgg16.trainable = False
    for l in my_vgg16.layers:
        l.trainable = False
    model = Model(inputs=my_vgg16.input, outputs=my_vgg16.get_layer('block4_conv3').output)
    model.trainable = False

    # preprocessing, since VGG16 works with data in the range of [0,255], so the images have to be reverted
    # the images are multiplied with the coverage-matrix in order to have all the non-covered pixels black
    yt_new = preprocess_to_caffe(revert_zero_center(y_true*covered_area)*255.0)
    yp_new = preprocess_to_caffe(revert_zero_center(y_pred*covered_area)*255.0)

    return SCALE * percentage_perceptual * K.mean(mean_squared_error(model(yt_new), model(yp_new)))
