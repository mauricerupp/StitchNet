import numpy as np
import tensorflow as tf
from losses import l1_loss
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *

"""
y_true = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target1.npy')[:,:,-3:]
print(y_true.shape)
y_true = np.concatenate([y_true[:,:,:-3], y_true[:,:,-3:]], axis=2)
print(y_true.shape)

"""
y_true1 = np.array([[[0, 0.2, 0.1, 1,1,1],[0.4, 0.2, 0.1,0,0,0]], [[0.3, 0.0, 0.5,1,1,1],[0.4, 0.4, 0.1,0,0,0]]])
y_true2 = np.array([[[0.11, 0.22, 0.33, 0,0,0],[0.33, 0.22, 0.11,0,0,0]], [[0.7, 0.70, 0.0,0,0,0],[0.1, 0.0, 0.2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[-0.5, 0.2, 0.1],[0.330, 0.220, 0.110]], [[0.3, 1.0, 0.5],[0.1, 0.0, 0.20]]])
y_pred2 = np.array([[[0.11, 0.22, 0.33],[0.33, 0.22, 0.11]], [[0.7, 0.70, 0.0],[0.1, 0.0, 0.2]]])
y_pred = np.array(((y_pred1, y_pred2)))


sess = tf.InteractiveSession()
loss = l1_loss.custom_loss(y_true, y_pred)
print(sess.run(loss))
sess.close()
"""
y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
"""
