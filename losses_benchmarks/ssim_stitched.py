from utilities import *
from tensorflow.python.ops.image_ops import ssim


def stitched_ssim(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: the ssim of two images neglecting which area is covered and not covered
    """
    y_true = y_true[:, :, :, :-3]
    # convert the images from [-1,1] to [0,1]
    y_pred = revert_zero_center(y_pred)
    y_true = revert_zero_center(y_true)

    return ssim(y_true, y_pred, max_val=1.0)



