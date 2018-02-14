from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
"""
Based on: https://gist.github.com/smly/d29d079100f8d81b905e
"""
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        if isinstance(self.X_val, pd.DataFrame):
            self.X_val = self.X_val.as_matrix()
        if isinstance(self.y_val, pd.DataFrame):
            self.y_val = self.y_val.as_matrix()

    def on_train_begin(self, logs={}):
        self.roc_scores = []
        self.conf_matrices = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            roc_score = roc_auc_score(self.y_val, y_pred)
            self.roc_scores.append(roc_score)
            print "Roc score: %.4f" % roc_score
            cm = confusion_matrix(self.y_val.argmax(axis=1), y_pred.argmax(axis=1))
            self.conf_matrices.append(cm)
            print "Confusion Matrix: "
            print cm
