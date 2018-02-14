from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_fscore_support
from tabulate import tabulate

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
        self.y_classes = []
        if isinstance(self.X_val, pd.DataFrame):
            self.X_val = self.X_val.as_matrix()
        if isinstance(self.y_val, pd.DataFrame):
            self.y_classes = list(self.y_val.columns.values)
            self.y_val = self.y_val.as_matrix()

    def on_train_begin(self, logs={}):
        self.roc_scores = []
        self.conf_matrices = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            self.rocScore(y_pred)
            self.confusionMatrix(y_pred)
            self.precisionRecallFScoreSupport(y_pred)

    def rocScore(self, y_pred):
        #roc_score = roc_auc_score(self.y_val, y_pred)
        #self.roc_scores.append(roc_score)
        #print "Roc score: %.4f" % roc_score
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(self.y_classes)):
            fpr[i], tpr[i], _ = roc_curve(self.y_val[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        print tabulate(
                [
                    [self.y_classes[0], roc_auc[0]],
                    [self.y_classes[1], roc_auc[1]],
                    [self.y_classes[2], roc_auc[2]]
                ],
                headers=["class", "roc_curve"],
                tablefmt="psql"
        )

    def confusionMatrix(self, y_pred):
        cm = confusion_matrix(
            self.y_val.argmax(axis=1),
            y_pred.argmax(axis=1)
        )
        self.conf_matrices.append(cm)
        print "Confusion Matrix: "
        if self.y_classes:
            print_cm(cm, self.y_classes, decimals=0)
        else:
            print cm
        print "Confusion Matrix - Normalized: "
        if self.y_classes:
            print_cm(cm.astype(float)/cm.sum(axis=1)[:, None], self.y_classes, decimals=3)
        else:
            print cm.astype(float)/cm.sum(axis=1)[:, None]

    def precisionRecallFScoreSupport(self, y_pred):
        predicted = [1,2,3,4,5,1,2,1,1,4,5]
        y_test = [1,2,3,4,5,1,2,1,1,4,1]

        precision, recall, fscore, support = precision_recall_fscore_support(
            self.y_val.argmax(axis=1),
            y_pred.argmax(axis=1)
        )

        print tabulate(
                [
                    [self.y_classes[0], precision[0], recall[0], fscore[0], support[0]],
                    [self.y_classes[1], precision[1], recall[1], fscore[1], support[1]],
                    [self.y_classes[2], precision[2], recall[2], fscore[2], support[2]]
                ],
                headers=["class", "precision", "recall", "fscore", "support"],
                tablefmt="psql"
            )

"""
https://gist.github.com/zachguo/10296432
"""
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, decimals=1):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.{1}f".format(columnwidth, decimals) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print
