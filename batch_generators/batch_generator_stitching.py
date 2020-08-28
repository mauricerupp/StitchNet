import math
from tensorflow.python.keras.utils import Sequence
from utilities import *
from S2_smooth_random_path_one_img import create_smooth_rand_path
from S1_fixed_path_one_img import create_fixed_path
from S3_very_random_path_one_img import create_very_rand_path


class MyGenerator(Sequence):
    """
    A batch generator which is used for all the presented networks.
    It returns batches of the indicated dataset which all have the form [x, y_true]
    x has the shape [64,64, amount_of_snapshots * 3]
    y_true has the shape [128,128,6], where the first 3 channels correspond to RGB and the last to the covered area
    The images are converted from [0,255] to [-1,1].
    """
    def __init__(self, raw_paths, batch_size, dataset="S2"):
        self.snaps = np.load(raw_paths)
        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self):
        return math.ceil(len(self.snaps) / float(self.batch_size))

    def __getitem__(self, idx):
        batch = self.snaps[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        targets = []
        for img_path in batch:
            if self.dataset == "S1":
                output = create_fixed_path(img_path)
            elif self.dataset == "S2":
                output = create_smooth_rand_path(img_path)
            else:
                output = create_very_rand_path(img_path)

            inputs.append(output[0])
            targets.append(output[1])

        return np.stack(inputs, axis=0), np.stack(targets, axis=0)
