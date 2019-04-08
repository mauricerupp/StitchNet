# own classes
import batch_generator
import u_net_superres_model2

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import time
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# set the constants
batchsize = 25
paths_dir_train = '/data/cvg/maurice/processed/coco/train'
paths_dir_val = '/data/cvg/maurice/processed/coco/val'
x_0 = np.load(paths_dir_train + "/snaps/snaps1.npy")
input_size = x_0.shape
x_0 = None

# name the model
NAME = "Image-Stitcher-u-net-superres2-{}".format(int(time.time()))

# make a TensorBoard
tensorboard = TensorBoard(log_dir='/data/cvg/maurice/logs/tb_logs/{}'.format(NAME))

# create checkpoint callbacks to store the training weights
checkpoint_path = '/data/cvg/maurice/logs/weight_logs/'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# create a batch generator
train_data_generator = batch_generator.MyGenerator(paths_dir_train + "/snaps_paths.npy",
                                                   paths_dir_train + "/targets_paths.npy", batchsize)
val_data_generator = batch_generator.MyGenerator(paths_dir_val + "/snaps_paths.npy",
                                                 paths_dir_val + "/targets_paths.npy", batchsize)

# setup the model
model = u_net_superres_model2.create_u_net_superres_model(input_size=input_size)

# train the model
model.fit_generator(train_data_generator,  epochs=10, callbacks=[cp_callback, tensorboard],
                    validation_data=val_data_generator, shuffle=True)

