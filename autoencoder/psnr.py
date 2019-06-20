import tensorflow.keras.backend as K
from utilities import *


def PSNR(y_true, y_pred):
    # convert the images back to [0,1]:
    y_pred = revert_zero_center(y_pred)
    y_true = revert_zero_center(y_true)
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
