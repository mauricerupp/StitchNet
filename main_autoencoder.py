# own classes
from batch_generator_autoencoder import *
from autoencoder_v6 import *
from utilities import *
from encoder_callback import *

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

#os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
tf.keras.backend.clear_session()

# set the constants
batchsize = 64
paths_dir = '/data/cvg/maurice/unprocessed/'
input_size = [64, 64, 3]
current_model = ConvAutoencoder

# name the model
NAME = str(current_model.__name__) + "_V6_instance_20_80_newcallback_run4"


# ----- Callbacks / Helperfunctions ----- #
def image_predictor(epoch, logs):
    """
    creates a tester, that predicts the same few images after every epoch and stores them as png
    we take 4 from the training and 4 from the validation set
    :param epoch:
    :param logs: has to be given as argument in order to compile
    """
    if epoch % 20 == 0:  # print samples every 50 images
        for i in range(1, 15):
            # load the ground truth
            set = ""
            if i % 2 == 0:
                list = np.load('/data/cvg/maurice/unprocessed/train_snaps_paths.npy')
                img = np.array(cv2.imread(list[i]))
                set += "train-"
            else:
                list = np.load('/data/cvg/maurice/unprocessed/val_snaps_paths.npy')
                img = np.array(cv2.imread(list[i]))
                set += "test-"

            # predict y (since the model is trained on pictures in [-1,1]) and we always take the same crop
            img = random_numpy_crop(img, input_size)
            y_true = np.expand_dims(img, axis=0)
            y_pred = model.autoencoder.predict(zero_center(y_true/255.0))
            y_pred = revert_zero_center(y_pred)*255.0
            y_pred = np.array(np.rint(y_pred), dtype=int)

            # save the result
            fig = plt.figure()
            fig.suptitle('Results of predicting {}Image {}\n on epoch {}'.format(set, i, epoch + 1), fontsize=20)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title('Y_True')
            plt.imshow(img[..., ::-1], interpolation='nearest') # conversion to RGB
            ax3 = fig.add_subplot(1, 2, 2)
            ax3.set_title('Prediction of model')
            plt.imshow(y_pred[0][..., ::-1], interpolation='nearest')
            plt.savefig("/data/cvg/maurice/logs/{}/Prediction-img{}-epoch{}.png".format(NAME, i, epoch + 1))
            plt.close()


cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)

# create a TensorBoard
tensorboard = TensorBoard(log_dir='/data/cvg/maurice/logs/{}/tb_logs/'.format(NAME), histogram_freq=0)


# ----- Batch-generator setup ----- #
train_data_generator = MyGenerator(paths_dir + "train_snaps_paths.npy", batchsize, input_size)
val_data_generator = MyGenerator(paths_dir + "val_snaps_paths.npy", batchsize, input_size)

# ----- Model setup ----- #
model = ConvAutoencoder(input_size, norm='instance', isTraining=True)
model.autoencoder.load_weights('/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback_run3/weight_logs/auto_weights-improvement-95.hdf5')

# create checkpoint callbacks to store the training weights
SAVE_PATH = '/data/cvg/maurice/logs/{}/weight_logs/auto'.format(NAME)
filepath = SAVE_PATH + '_weights-improvement-{epoch:02d}.hdf5'
cp_callback = keras.callbacks.ModelCheckpointv


# train the model
model.autoencoder.fit_generator(train_data_generator,  epochs=800,
                    callbacks=[cp_callback, tensorboard, cb_imagepredict],
                    validation_data=val_data_generator, max_queue_size=64, workers=12)
