import tensorflow.keras.backend as K
import tensorflow as tf


def my_loss_l1(y_true, y_pred):
    """
    Idea from paper perceptual losses for real.time style transfer and super-resolution

    :return: the loss of covered pixels
    """
    shape = y_pred.shape
    mod_vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                  input_shape=shape)
    features = mod_vgg16.predict(y_pred)
    covered_area = y_true[:, :, :, -3:]
    y_true = y_true[:, :, :, :-3]
    return K.sum(K.abs(y_true - y_pred) * covered_area)
