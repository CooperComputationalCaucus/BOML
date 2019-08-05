'''
Created on 17 Apr 2019

@author: pmm
'''
from keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class ClassificationMetrics(Callback):
    # TODO: Paralelize
    def __init__(self):
        super().__init__()
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(self.validation_data[0]))), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict, axis=0))
        _val_recall = recall_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict, axis=0))
        _val_precision = precision_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict, axis=0))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print(“ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall))
        return


class GeneratorClassificationMetrics(Callback):
    # TODO: Paralelize
    def __init__(self, validation_generator, validation_steps):
        super().__init__()
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = None
        val_targ = None
        for idx in range(self.validation_steps):
            features, targ = self.validation_generator.__getitem__(idx)
            if val_pred is None:
                val_pred = np.argmax(np.asarray(self.model.predict(features)), axis=1)
                # val_pred = val_pred.round().astype(int)
                val_targ = np.argmax(targ, axis=1)
            else:
                pred = np.argmax(np.asarray(self.model.predict(features)), axis=1)
                # pred = pred.round().astype(int)
                val_pred = np.append(val_pred, pred, axis=0)
                val_targ = np.append(val_targ, np.argmax(targ, axis=1), axis=0)
        _val_f1 = f1_score(val_targ, val_pred, average='macro', labels=np.unique(val_pred))
        _val_recall = recall_score(val_targ, val_pred, average='macro', labels=np.unique(val_pred))
        _val_precision = precision_score(val_targ, val_pred, average='macro', labels=np.unique(val_pred))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return


class RegressionMetrics(Callback):
    # TODO: Paralelize
    def __init__(self):
        super().__init__()
        self.r2 = []
        self.mse = []

    def on_train_begin(self, logs={}):
        self.r2 = []
        self.mse = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]
        self.r2.append(r2_score(val_predict, val_targ))
        self.mse.append(mean_squared_error(val_predict, val_targ))
        return


class GeneratorRegressionMetrics(Callback):
    # TODO: Paralelize
    def __init__(self, validation_generator, validation_steps):
        super().__init__()
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.r2 = []
        self.mse = []

    def on_train_begin(self, logs={}):
        self.r2 = []
        self.mse = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = None
        for features, targ in self.validation_generator:
            if val_pred is None:
                val_pred = np.asarray(self.model.predict(features))
                val_targ = targ
            else:
                pred = np.argmax(np.asarray(self.model.predict(features)), axis=1)
                val_pred = np.append(val_pred, pred, axis=0)
                val_targ = np.append(val_targ, targ, axis=0)
        self.r2.append(r2_score(val_pred, val_targ))
        self.mse.append(mean_squared_error(val_pred, val_targ))
        return