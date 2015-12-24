from keras.callbacks import Callback
import numpy as np
import pickle

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss',
                 smaller_is_better=True,
                 patience=0, verbose=0):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = np.Inf
        self.smaller_is_better = smaller_is_better
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.smaller_is_better is False:
            current = -current

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_attr_dict(d):
    d_ = AttrDict()
    d_.update(d)
    return d_


class LearningRateScheduler(Callback):
    '''Learning rate scheduler.
    # Arguments
        schedule: a function that gets an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output.
    '''
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
        self.model.optimizer.lr.set_value(self.schedule(epoch, self.model.optimizer.lr.get_value()))
