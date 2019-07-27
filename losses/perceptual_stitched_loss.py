from tensorflow.python.keras.applications.vgg16 import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model
from utilities import *
from tensorflow.python.keras.losses import *
from l2_loss import my_loss_l2
from l1_loss import custom_loss


def vgg_loss(y_true, y_pred):
    """
    does a weighted loss of perceptual loss and MAE
    :param y_true:
    :param y_pred:
    :return:
    """
    percentage_MAE = 0.2
    percentage_perceptual = 1 - percentage_MAE
    SCALE = 5.57779e-05
    covered_area = y_true[:, :, :, -3:]

    # MAE has shape of (64x64), Perceptual has shape of (8x8), therefore we take the mean of those values and
    # add a scale factor, so they have the same general scale, so they can be weighted properly
    return percentage_MAE * custom_loss(y_true, y_pred) + \
           SCALE * percentage_perceptual * perceptual_loss(y_true[:, :, :, :-3], y_pred, covered_area)


def perceptual_loss(y_true, y_pred, covered_area):
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
    # also the images are multiplied with the coverage-matrix in order to have all the non-covered pixels black
    #TODO: multiply it with the covered area before or after the prediction? --> Then maybe change SCALE
    yt_new = preprocess_to_caffe(revert_zero_center(y_true*covered_area)*255.0)
    yp_new = preprocess_to_caffe(revert_zero_center(y_pred*covered_area)*255.0)
    # since we here have 8x8=64 pixels, we have to scale the result
    return K.mean(mean_squared_error(model(yt_new), model(yp_new)))
