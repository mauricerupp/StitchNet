import tensorflow.keras.backend as K
import tensorflow as tf


def custom_loss(y_true, y_pred):
    """
    first takes the element-wise absolute value of true and predicted values
    and then multiplies this with the coverage-matrix in order to weight covered pixels with 1 and
    non-covered ones with 0
    :return: the loss of covered pixels
    """
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]
    l1 = K.sum(K.abs(y_true - y_pred) * covered_area)
    nonzero = K.cast(tf.math.count_nonzero(covered_area, keepdims=False), 'float32')
    # get the mean absolute error, but the value of the mean is only the actually covered pixels
    return l1/nonzero