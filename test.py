import numpy as np
import math
from batch_generator import MyGenerator
import tensorflow.keras.backend as K


y_pred = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/target_arrays/img_target1.npy')
print(y_pred.shape)
y = y_pred[:,:,:-3]
print(y.shape)