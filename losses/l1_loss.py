import tensorflow.keras.backend as K
import numpy as np


def revert_zero_center(in_img):
    return in_img / 2 + 0.5


def custom_loss(y_true, y_pred):

    """
    first takes the element-wise absolute value of true and predicted values
    and then multiplies this with the coverage-matrix in order to weight covered pixels with 1 and
    non-covered ones with 0
    :return: the loss of covered pixels
    """
    covered_area = y_true[:, :, :, -3:]
    print(covered_area)
    y_true = y_true[:, :, :, :-3]
    return K.sum(K.abs(y_true - y_pred) * covered_area)
