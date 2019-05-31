import numpy as np
import os

dir = '/data/cvg/maurice/unprocessed/coco_smalltrain/'
snaps_paths = []
storage_dir = '/data/cvg/maurice/unprocessed/'
for img in os.listdir(dir):
    snaps_paths.append(os.path.join(dir, img))

np.save(storage_dir + "train_snaps_paths.npy", snaps_paths)


