import numpy as np
import math
import l1_loss
import tensorflow.keras.backend as K
import tensorflow as tf
import u_net_convtrans_model2
import u_net_convtrans_model3
import cv2
import scipy.misc
import os

dir = '/home/maurice/Dokumente/Try_Models/coco_try/TR'
max_h = 0
max_w = 0
min_h = 1000000
min_w = 1000000
small_h_counter = 0
small_w_counter = 0
for img in os.listdir(dir):
    img_target = cv2.imread(os.path.join(dir, img))
    (h, w) = img_target.shape[:2]
    if h > w:
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 90, 1)
        img_target = cv2.warpAffine(img_target, M, (h, w))
    temp = h
    h = w
    w = h
    if h > max_h:
        max_h = h
    if w > max_w:
        max_w = w

    if h < min_h:
        min_h = h
    if w < min_w:
        min_w = w


for img in os.listdir(dir):
    img_target = cv2.imread(os.path.join(dir, img))
    (h, w) = img_target.shape[:2]
    if h > w:
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 90, 1)
        img_target = cv2.warpAffine(img_target, M, (h, w))
    temp = h
    h = w
    w = h
    if h < min_h + 100:
        small_h_counter += 1
    if w < min_w + 100:
        small_w_counter += 1



print(max_h, max_w)
print(min_h, min_w)
print(small_h_counter, small_w_counter)

"""
y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
y_true2 = np.array([[[11, 22, 33, 0,0,0],[33, 22, 11,0,0,0]], [[7, 70, 0,0,0,0],[1, 0, 2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[1, 2, 1],[330, 220, 110]], [[3, 0, 5],[1, 0, 20]]])
y_pred2 = np.array([[[100, 200, 100],[330, 220, 110]], [[3, 0, 5],[0, 1, 2]]])
y_pred = np.array(((y_pred1, y_pred2)))

sess = tf.InteractiveSession()
loss = loss.my_loss_l1(y_true, y_pred)
print(sess.run(loss))
print(loss.eval())
sess.close()




y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
np.load()
"""
