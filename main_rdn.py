# own classes
from batch_generator_stitching import *
from rdn import *
from S1_fixed_path_one_img import *
from S2_smooth_random_path_one_img import *
from S3_very_random_path_one_img import *
from utilities import image_predictor

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--inputdir", type=str, default='/data/cvg/maurice/unprocessed/')
    parser.add_argument("--storagedir", type=str, default='/data/cvg/maurice/logs/')
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--C", type=int, default=6)
    parser.add_argument("--G0", type=int, default=320)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--datasettype", type=str, default='S1')
    args = parser.parse_args()
    
    tf.keras.backend.clear_session()
    
    # set the constants
    batchsize = args.batchsize
    paths_dir = args.inputdir
    input_size = [64,64,15]
    
    # name the model and choose the dataset
    DATASET = args.datasettype
    NAME = "RDN_" + DATASET
    
    # ----- Callbacks ----- #
    cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)
    SAVE_PATH = args.storagedir + '{}/weight_logs/rdn'.format(NAME)
    filepath = SAVE_PATH + '_weights-improvement-{epoch:02d}.hdf5'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=10, save_weights_only=True)
    
    # create a TensorBoard
    tensorboard = TensorBoard(log_dir=args.storagedir + '{}/tb_logs/'.format(NAME))
    
    # ----- Batch-generator setup ----- #
    train_data_generator = MyGenerator(paths_dir + "train_snaps_paths.npy", batchsize, DATASET)
    val_data_generator = MyGenerator(paths_dir + "val_snaps_paths.npy", batchsize, DATASET)
    
    # ----- Model setup ----- #
    model = create_model(input_size=input_size, D=args.D, C=args.C, G0=args.G0, G=args.G)
    
    # ----- Training ----- #
    model.fit_generator(train_data_generator,  epochs=args.epochs,
                        callbacks=[cp_callback, tensorboard, cb_imagepredict],
                        validation_data=val_data_generator, max_queue_size=64, workers=12)


if __name__ == '__main__':
    main()
