from tensorflow.python.keras.callbacks import *


class EncoderCheckpoint(Callback):
    def __init__(self, filepath, encoder):
        self.monitor = 'val_loss'
        self.monitor_op = np.less
        self.best = np.Inf

        self.filepath = filepath
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.encoder.save_weights(self.filepath + '{}.ckpt'.format(epoch), overwrite=True)