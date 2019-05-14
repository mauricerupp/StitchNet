import numpy as np
import math
from tensorflow.python.keras.utils import Sequence


# ---- Helpers ---- #
def zero_center(in_img):
    return 2 * in_img - 1


def revert_zero_center(in_img):
    return in_img / 2 + 0.5
# ---- END ---- #


class MyGenerator(Sequence):

    def __init__(self, snaps_paths, targets_paths, batch_size):
        self.snaps, self.targets = np.load(snaps_paths), np.load(targets_paths)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        print(np.array([np.load(img_name)/255.0 for img_name in batch_x]))
        return np.array([zero_center(np.load(img_name)/255.0) for img_name in batch_x]), \
            np.array([zero_center(np.load(target_name)/255.0) for target_name in batch_y])

