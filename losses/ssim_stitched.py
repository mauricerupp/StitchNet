import tensorflow.keras.backend as K
from utilities import *
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.image_ops import ssim


def stitched_ssim(y_true, y_pred):
    # calculates the PSNR for the covered pixels
    # convert the images back to [0,1]:
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]
    # from [-1,1] to [0,1]
    y_pred = revert_zero_center(y_pred)
    y_true = revert_zero_center(y_true)

    return ssim(y_true, y_pred, max_val=1.0)



