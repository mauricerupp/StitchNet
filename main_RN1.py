# own classes
from batch_generator import *
from RN_2 import *
from utilities import *
from fixed_path_one_img import *
from smooth_random_path_one_img import *
from very_random_path_one_img import *

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# set the constants
batchsize = 48
paths_dir = '/data/cvg/maurice/unprocessed/'
input_size = [64,64,15]

# name the model
DATASET = "S2"
NAME = "RN1_" + DATASET


# ----- Callbacks / Helperfunctions ----- #
def image_predictor(epoch, logs):
    """
    createy a tester, that predicts the same few images after every epoch and stores them as png
    we take 4 from the training and 4 from the validation set
    :param epoch:
    :param logs: has to be given as argument in order to compile
    """
    if epoch % 2 == 0:  # print samples every 50 images
        for i in range(0,25):
            # load X
            set = ""
            if i % 10 == 0:
                list = np.load('/data/cvg/maurice/unprocessed/train_snaps_paths.npy')
                set += "train-"
            else:
                list = np.load('/data/cvg/maurice/unprocessed/val_snaps_paths.npy')
                set += "test-"

            # create a random path
            if DATASET == "S1":
                loaded_data = create_fixed_path(list[i])
            elif DATASET == "S2":
                loaded_data = create_smooth_rand_path(list[i])
            else:
                loaded_data = create_very_rand_path(list[i])

            # preprocess x
            x = loaded_data[0]
            x = np.expand_dims(x, axis=0)
            # preprocess y
            y_true = loaded_data[1]
            covered_area = y_true[:, :, -3:]
            y_true = y_true[:, :, :-3]
            covered_target = y_true * covered_area

            # predict y (since the model is trained on pictures in [-1,1])
            y_pred = model.predict(x)
            y_pred = revert_zero_center(y_pred)*255
            y_pred = np.array(np.rint(y_pred), dtype=int)

            # save the result
            fig = plt.figure()
            fig.suptitle('Results of predicting {}Image {} \non epoch {}'.format(set, i, epoch + 1), fontsize=20)
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('Y_True')
            plt.imshow(y_true[..., ::-1], interpolation='nearest') # conversion to RGB
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('Y_True covered')
            plt.imshow(covered_target[..., ::-1], interpolation='nearest')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('Prediction of model')
            plt.imshow(y_pred[0][..., ::-1], interpolation='nearest')
            plt.savefig("/data/cvg/maurice/logs/{}/Prediction-img{}-epoch{}.png".format(NAME, i, epoch + 1))
            plt.close()


cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)

# create a TensorBoard
tensorboard = TensorBoard(log_dir='/data/cvg/maurice/logs/{}/tb_logs/'.format(NAME))

# create checkpoint callbacks to store the training weights
SAVE_PATH = '/data/cvg/maurice/logs/{}/weight_logs/rn1'.format(NAME)
filepath = SAVE_PATH + '_weights-improvement-{epoch:02d}.hdf5'
cp_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=1, save_weights_only=True)

# ----- Batch-generator setup ----- #
train_data_generator = MyGenerator(paths_dir + "train_snaps_paths.npy", batchsize, DATASET)
val_data_generator = MyGenerator(paths_dir + "val_snaps_paths.npy", batchsize, DATASET)

# ----- Model setup ----- #
model = create_model(input_size=input_size, block_amount=20, normalizer="instance", filter_size=128)
#model.load_weights('/data/cvg/maurice/logs/StitchDecoder_AEv6_D2v4_MAE/weight_logs/')

# train the model
model.fit_generator(train_data_generator,  epochs=702,
                    callbacks=[cp_callback, tensorboard, cb_imagepredict],
                    validation_data=val_data_generator, max_queue_size=64, workers=12)

