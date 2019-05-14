import numpy as np
import math
from tensorflow.python.keras.utils import Sequence


class MyGenerator(Sequence):

    def __init__(self, snaps_paths, targets_paths, batch_size):
        self.snaps, self.targets = np.load(snaps_paths), np.load(targets_paths)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.load(img_name)/255.0 for img_name in batch_x]), \
            np.array([np.load(target_name)/255.0 for target_name in batch_y])
