# own classes
from batch_generator import *
from stitch_decoder import *
from utilities import *

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
tf.keras.backend.clear_session()

# set the constants
batchsize = 400
paths_dir_train = '/data/cvg/maurice/processed/coco_small/train'
paths_dir_val = '/data/cvg/maurice/processed/coco_small/val'
x_0 = np.load(paths_dir_train + "/snaps/snaps1.npy")
input_size = x_0.shape
x_0 = None
current_model = StitchDecoder

# name the model
NAME = str(current_model.__name__) + "_v2_small_coco_LoadedAutoWeights"


# ----- Callbacks / Helperfunctions ----- #
def image_predictor(epoch, logs):
    """
    createy a tester, that predicts the same few images after every epoch and stores them as png
    we take 4 from the training and 4 from the validation set
    :param epoch:
    :param logs: has to be given as argument in order to compile
    """
    if epoch % 250 == 0:  # print samples every 50 images
        for i in range(1, 5):
            # load X
            if i % 2 == 0:
                x_pred = np.load('/data/cvg/maurice/processed/coco_small/train/snaps/snaps{}.npy'.format(i))
            else:
                x_pred = np.load('/data/cvg/maurice/processed/coco_small/val/snaps/snaps{}.npy'.format(i))
            x_pred = np.expand_dims(x_pred, axis=0)

            # load Y
            if i % 2 == 0:
                y_true = np.load('/data/cvg/maurice/processed/coco_small/train/targets/target{}.npy'.format(i))
            else:
                y_true = np.load('/data/cvg/maurice/processed/coco_small/val/targets/target{}.npy'.format(i))
            covered_area = y_true[:, :, -3:]
            y_true = y_true[:, :, :-3]
            covered_target = y_true * covered_area

            # predict y (since the model is trained on pictures in [-1,1])
            y_pred = model.stitchdecoder.predict(zero_center(x_pred/255.0))
            equality = np.equal(y_pred, zero_center(y_true / 255.0))
            accuracy = np.mean(equality)
            y_pred = revert_zero_center(y_pred)*255.0
            y_pred = np.array(np.rint(y_pred), dtype=int)

            # save the result
            fig = plt.figure()
            fig.suptitle('Results of predicting Image {} on epoch {} \nwith an accuracy of {:.2%}'.format(i, epoch + 1, accuracy), fontsize=20)
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
checkpoint_path = '/data/cvg/maurice/logs/{}/weight_logs/'.format(NAME)
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# ----- Batch-generator setup ----- #
train_data_generator = MyGenerator(paths_dir_train + "/snaps_paths.npy", paths_dir_train + "/targets_paths.npy", batchsize, '-1,1')
val_data_generator = MyGenerator(paths_dir_val + "/snaps_paths.npy", paths_dir_val + "/targets_paths.npy", batchsize, '-1,1')

# ----- Model setup ----- #
model = StitchDecoder(input_size, '/data/cvg/maurice/logs/ConvAutoencoder_V2_run7/weight_logs/')

# train the model
model.stitchdecoder.fit_generator(train_data_generator,  epochs=4002,
                    callbacks=[cp_callback, tensorboard, cb_imagepredict],
                    validation_data=val_data_generator)

