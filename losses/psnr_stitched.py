import tensorflow.keras.backend as K
from utilities import *


def stitched_PSNR(y_true, y_pred):
    # calculates the PSNR for the covered pixels
    # convert the images back to [0,1]:
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]
    y_pred = revert_zero_center(y_pred)
    y_true = revert_zero_center(y_true)
    max_pixel = 1.0
    nonzero = K.cast(tf.math.count_nonzero(covered_area, keepdims=False), 'float32')
    return 10.0 * log10((max_pixel ** 2) / (K.sum(K.square(y_pred - y_true) * covered_area)/nonzero))
