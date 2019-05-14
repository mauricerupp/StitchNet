import tensorflow.keras.backend as K
import numpy as np


def revert_zero_center(in_img):
    return in_img / 2 + 0.5


def custom_loss(pixelvalue_range):
    """
    first takes the element-wise absolute value of true and predicted values
    and then multiplies this with the coverage-matrix in order to weight covered pixels with 1 and
    non-covered ones with 0
    Needs a switch-statement, since the covered area is part of the batch-generator and can therefore have values,
    which are not in (0,1)
    :return: the loss of covered pixels
    """
    def lossFunction(y_true, y_pred):
        if pixelvalue_range == '0_255':
            covered_area = y_true[:, :, :, -3:]
        elif pixelvalue_range == '0_1':
            covered_area = np.array(y_true[:, :, :, -3:] * 255, dtype=int)
        elif pixelvalue_range == 'minus1_1':
            covered_area = np.array(revert_zero_center(y_true[:, :, :, -3:]) * 255, dtype=int)
        else:
            print('no valuable pixelvalue range indicated')
            exit()
        print(covered_area)
        y_true = y_true[:, :, :, :-3]
        return K.sum(K.abs(y_true - y_pred) * covered_area)

    return lossFunction
