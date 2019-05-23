import numpy as np
import math
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input


class MyGenerator(Sequence):

    def __init__(self, snaps_paths, targets_paths, batch_size, mode):
        self.snaps, self.targets = np.load(snaps_paths), np.load(targets_paths)
        self.batch_size = batch_size
        self.mode = mode

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx, mode):
        batch_x = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        # depending on which range we want to train
        if mode.lower() == '0,1':
            return np.array([np.load(img_name)/255.0 for img_name in batch_x]), \
                   np.array([np.concatenate([np.load(target_name)[:, :, :-3]/255.0,
                                             np.load(target_name)[:, :, -3:]], axis=2) for target_name in batch_y])
        elif mode.lower() == '-1,1':
            return np.array([np.load(img_name)/255.0 for img_name in batch_x]), \
                   np.array([np.concatenate([preprocess_input(np.load(target_name)[:, :, :-3], mode='tf'),
                                             np.load(target_name)[:, :, -3:]], axis=2) for target_name in batch_y])
        else:
            return np.array([np.load(img_name) for img_name in batch_x]), \
                   np.array([np.load(target_name) for target_name in batch_y])
