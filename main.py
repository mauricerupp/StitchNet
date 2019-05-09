# own classes
import batch_generator
import u_net_convtrans_model2_FULLBATCHNORM

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# set the constants
batchsize = 70
paths_dir_train = '/data/cvg/maurice/processed/coco/train'
paths_dir_val = '/data/cvg/maurice/processed/coco/val'
x_0 = np.load(paths_dir_train + "/snaps/snaps1.npy")
input_size = x_0.shape
x_0 = None
current_model = u_net_convtrans_model2_FULLBATCHNORM    

# name the model
NAME = str(current_model.__name__)

# create a TensorBoard
tensorboard = TensorBoard(log_dir='/data/cvg/maurice/logs/{}/tb_logs/'.format(NAME))

# create checkpoint callbacks to store the training weights
checkpoint_path = '/data/cvg/maurice/logs/{}/weight_logs/'.format(NAME)
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


# create a tester, that predicts the same few images after every epoch and stores them as png
# we take 4 from the training and 4 from the validation set
def image_predictor(epoch, logs):
    for i in range(1,15):
        # load X
        if i%2 == 0:
            x_pred = np.load('/data/cvg/maurice/processed/coco/train/snaps/snaps{}.npy'.format(i))
        else:
            x_pred = np.load('/data/cvg/maurice/processed/coco/val/snaps/snaps{}.npy'.format(i))
        x_pred = np.expand_dims(x_pred, axis=0)
        # since we train the model with values between 0 and 1 now
        x_pred = x_pred

        # load Y
        if i%2 == 0:
            y_true = np.load('/data/cvg/maurice/processed/coco/train/targets/target{}.npy'.format(i))
        else:
            y_true = np.load('/data/cvg/maurice/processed/coco/val/targets/target{}.npy'.format(i))
        covered_area = y_true[:, :, -3:]
        y_true = y_true[:, :, :-3]
        covered_target = y_true * covered_area

        # predict y
        y_pred = model.predict(x_pred)
        y_pred = np.array(np.rint(y_pred), dtype=int)

        # save the result
        fig = plt.figure()
        fig.suptitle('Results of predicting Image {} on epoch {}'.format(i, epoch + 1), fontsize=20)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Y_True')
        plt.imshow(y_true[..., ::-1], interpolation='nearest')
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Y_True covered')
        plt.imshow(covered_target[..., ::-1], interpolation='nearest')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Prediction of model')
        plt.imshow(y_pred[0][..., ::-1], interpolation='nearest')
        plt.savefig("/data/cvg/maurice/logs/{}/Prediction-img{}-epoch{}.png".format(NAME, i, epoch + 1))


cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)

# create a batch generator
train_data_generator = batch_generator.MyGenerator(paths_dir_train + "/snaps_paths.npy",
                                                   paths_dir_train + "/targets_paths.npy", batchsize)
val_data_generator = batch_generator.MyGenerator(paths_dir_val + "/snaps_paths.npy",
                                                 paths_dir_val + "/targets_paths.npy", batchsize)

# setup the model
model = current_model.create_model(input_size=input_size)

# train the model
model.fit_generator(train_data_generator,  epochs=15,
                    callbacks=[cp_callback, tensorboard, cb_imagepredict],
                    validation_data=val_data_generator)

