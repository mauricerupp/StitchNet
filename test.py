import numpy as np
import tensorflow as tf
from losses import l1_loss
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


y_true = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target1.npy')
print(y_true.shape)
y_true = np.concatenate([y_true[:,:,:-3], y_true[:,:,-3:]], axis=2)
print(y_true.shape)

"""
y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
y_true2 = np.array([[[11, 22, 33, 0,0,0],[33, 22, 11,0,0,0]], [[7, 70, 0,0,0,0],[1, 0, 2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[1, 2, 1],[330, 220, 110]], [[3, 0, 5],[1, 0, 20]]])
y_pred2 = np.array([[[100, 200, 100],[330, 220, 110]], [[3, 0, 5],[0, 1, 2]]])
y_pred = np.array(((y_pred1, y_pred2)))

sess = tf.InteractiveSession()
loss = l1_loss.custom_loss(pixelvalue_range='0_255')
print(sess.run(loss))
print(loss.eval())
sess.close()

y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
"""
