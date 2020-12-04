# own classes
from batch_generator_stitching import *
from stitchnet2 import *
from utilities import *

# packages
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--inputdir", type=str, default='/data/cvg/maurice/unprocessed/')
    parser.add_argument("--storagedir", type=str, default='/data/cvg/maurice/logs/')
    parser.add_argument("--epochs", type=int, default=5500)
    parser.add_argument("--datasettype", type=str, default='S1')
    parser.add_argument("--norm", type=str, default='instance')
    args = parser.parse_args()
    
    tf.keras.backend.clear_session()
    
    # set the constants
    batchsize = args.batchsize
    paths_dir = args.inputdir
    input_size = [64,64,15]
    current_model = StitchDecoder
    
    # name the model and choose the dataset
    DATASET = args.datasettype
    NAME = str(current_model.__name__) + "v1_" + DATASET
    
    
    cb_imagepredict = keras.callbacks.LambdaCallback(on_epoch_end=image_predictor)
    
    # create a TensorBoard
    tensorboard = TensorBoard(log_dir=args.storagedir + '{}/tb_logs/'.format(NAME))
    
    # create checkpoint callbacks to store the training weights
    SAVE_PATH = args.storagedir + '{}/weight_logs/d2'.format(NAME)
    filepath = SAVE_PATH + '_weights-improvement-{epoch:02d}.hdf5'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=5, save_weights_only=True)
    
    # ----- Batch-generator setup ----- #
    train_data_generator = MyGenerator(paths_dir + "train_snaps_paths.npy", batchsize, DATASET)
    val_data_generator = MyGenerator(paths_dir + "val_snaps_paths.npy", batchsize, DATASET)
    
    # ----- Model setup ----- #
    model = StitchDecoder(input_size, normalizer=args.norm, isTraining=True)
    
    # train the model
    model.stitchdecoder.fit_generator(train_data_generator,  epochs=args.epochs,
                        callbacks=[cp_callback, tensorboard, cb_imagepredict],
                        validation_data=val_data_generator, max_queue_size=64, workers=12)
    
    model.stitchdecoder.save(SAVE_PATH)


if __name__ == '__main__':
    main()
