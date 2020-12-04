# own classes
from batch_generator_autoencoder import *
from autoencoder_64x64 import *
from utilities import *

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--inputdir", type=str, default='/data/cvg/maurice/unprocessed/')
    parser.add_argument("--storagedir", type=str, default='/data/cvg/maurice/logs/')
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--norm", type=str, default='instance')
    args = parser.parse_args()
    
    tf.keras.backend.clear_session()
    
    # set the constants
    batchsize = args.batchsize
    paths_dir = args.inputdir
    input_size = [64, 64, 3]
    current_model = ConvAutoencoder
    
    # name the model
    NAME = str(current_model.__name__)
    
    
    # ----- Callbacks ----- #
    cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)
    SAVE_PATH = args.storagedir + '{}/weight_logs/auto'.format(NAME)
    filepath = SAVE_PATH + '_weights-improvement-{epoch:02d}.hdf5'
    cp_callback = keras.callbacks.ModelCheckpoint
    
    # create a TensorBoard
    tensorboard = TensorBoard(log_dir=args.storagedir + '{}/tb_logs/'.format(NAME), histogram_freq=0)
    
    # ----- Batch-generator setup ----- #
    train_data_generator = MyGenerator(paths_dir + "train_snaps_paths.npy", batchsize, input_size)
    val_data_generator = MyGenerator(paths_dir + "val_snaps_paths.npy", batchsize, input_size)
    
    # ----- Model setup ----- #
    model = ConvAutoencoder(input_size, norm=args.norm, isTraining=True)
    
    # ----- Training ----- #
    model.autoencoder.fit_generator(train_data_generator,  epochs=args.epochs,
                        callbacks=[cp_callback, tensorboard, cb_imagepredict],
                        validation_data=val_data_generator, max_queue_size=64, workers=12)


if __name__ == '__main__':
    main()
