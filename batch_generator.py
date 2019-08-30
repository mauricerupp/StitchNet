import numpy as np
import math
from tensorflow.python.keras.utils import Sequence
from utilities import *
from smooth_random_path_one_img import create_smooth_rand_path
from fixed_path_one_img import create_fixed_path
from very_random_path_one_img import create_very_rand_path


class MyGenerator(Sequence):

    def __init__(self, raw_paths, batch_size):
        self.snaps = np.load(raw_paths)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx):
        batch = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        targets = []
        for img_path in batch:
            output = create_smooth_rand_path(img_path)

            inputs.append(output[0])
            targets.append(output[1])

        return np.stack(inputs, axis=0), np.stack(targets, axis=0)
