import numpy as np
import math
from tensorflow.python.keras.utils import Sequence
from utilities import *
from tensorflow._api.v1.image import random_crop
import cv2


class MyGenerator(Sequence):

    def __init__(self, snaps_paths, batch_size, img_size):
        self.snaps = np.load(snaps_paths)
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx):
        batch = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([random_crop(zero_center(np.array(cv2.imread(img_name))/255.0), self.img_size) for img_name in batch]),\
               np.array([random_crop(zero_center(np.array(cv2.imread(img_name))/255.0), self.img_size) for img_name in batch])
