from tensorflow.python.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model
from utilities import *


def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=[64,64,3])
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    # since vgg was trained with images in [0,1] we revert it here from [-1,1]
    for i in y_true:
        y_true[i] = revert_zero_center(y_true[i])
        y_pred[i] = revert_zero_center(y_pred[i])
    return K.mean(K.square(model(y_true) - model(y_pred)))
