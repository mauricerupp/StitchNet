import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


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
    for i in range(y_true.shape[0]):
        temp = K.abs(y_true[i] - y_pred[i]) * covered_area[i]
        fig = plt.figure()
        plt.imshow(temp[..., ::-1], interpolation='nearest')
        plt.savefig("/data/cvg/maurice/logs/l1_masks/Lossmask-img{}.png".format(i))
    nonzero = K.cast(tf.math.count_nonzero(covered_area, keepdims=False), 'float32')
    # get the mean absolute error, but the value of the mean is only the actually covered pixels
    return l1/nonzero
