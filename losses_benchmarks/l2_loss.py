import tensorflow.keras.backend as K
import tensorflow as tf


def my_loss_l2(y_true, y_pred):
    """
    first takes the element-wise squared value of true and predicted values
    and then multiplies this with the coverage-matrix in order to weight covered pixels with 1 and
    non-covered ones with 0
    :return: the loss of covered pixels
    """
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]
    l2 = K.sum(K.pow(y_true - y_pred, 2) * covered_area)
    nonzero = K.cast(tf.math.count_nonzero(covered_area, keepdims=False), 'float32')

    return l2/nonzero
