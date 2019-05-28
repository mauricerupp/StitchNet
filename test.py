import numpy as np
import tensorflow as tf
from losses import l1_loss
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import l1_loss
import random

from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def alg(k):
    x = 0
    y = 2**k

    while x < y-1:
        r = random.randint(x, y-1)
        m = x + (y-x)/2
        if m <= r:
            x = m
        else:
            y = m
    return y

print(alg(2))
"""        
y_pred1 = np.array([[[0, 255, 2],[33, 22, 11]], [[0, 0, 222],[1, 0, 2]]])
print(y_pred1)
print(y_pred1.shape)
y_pred1 = preprocess_input(y_pred1, mode='torch')
print(y_pred1)
print(y_pred1.shape)

def create_model(pretrained_weights=None, input_size=None, G0=64, G=32, D=20, C=6):
    
    A simple residual network with ResBlocks from Givi
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    inputs = Input(input_size)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='global_conv2')(inputs)
    conv = create_resblock(conv, 'dx', 128, 3, 1, 1, BatchNormalization)
    model = Model(inputs=inputs, outputs=conv)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()


def create_resblock(prior_layer, block_name, n_filters, kernel_size, stride, dilation, normalizer):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', name='global_conv3')(prior_layer)
    x = normalizer()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation, name=block_name + "conv2")(x)

    x = normalizer()(x)

    return x
    #return Add()([prior_layer, x])


mod = create_model(input_size=(64,64,15))

y_true = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target1.npy')[:,:,-3:]
print(y_true.shape)
y_true = np.concatenate([y_true[:,:,:-3], y_true[:,:,-3:]], axis=2)
print(y_true.shape)

y_true1 = np.array([[[0, 0.2, 0.1, 1,1,1],[0.4, 0.2, 0.1,0,0,0]], [[0.3, 0.0, 0.5,1,1,1],[0.4, 0.4, 0.1,0,0,0]]])
y_true2 = np.array([[[0.11, 0.22, 0.33, 0,0,0],[0.33, 0.22, 0.11,0,0,0]], [[0.7, 0.70, 0.0,0,0,0],[0.1, 0.0, 0.2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[-0.5, 0.2, 0.1],[0.330, 0.220, 0.110]], [[0.3, 1.0, 0.5],[0.1, 0.0, 0.20]]])
y_pred2 = np.array([[[0.11, 0.22, 0.33],[0.33, 0.22, 0.11]], [[0.7, 0.70, 0.0],[0.1, 0.0, 0.2]]])
y_pred = np.array(((y_pred1, y_pred2)))


sess = tf.InteractiveSession()
loss = l1_loss.custom_loss(y_true, y_pred)
equality = tf.equal(y_pred1, y_pred2 / 255.0)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

print(accuracy.eval())
print(sess.run(loss))
sess.close()

equality = np.equal(y_pred1, y_pred2 / 255.0)
accuracy = np.mean(equality)

print(accuracy)
print('Results of predicting Image {} on epoch {} with an accuracy of {:.2%}'.format(20, 1 + 1, accuracy))

y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
"""
