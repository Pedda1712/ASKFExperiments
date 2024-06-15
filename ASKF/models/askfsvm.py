import numpy as np
from ..models.askfsvm_binary import ASKFSVMBinary
from ..models.multiclass_strategy import OneVsRestClassifier
from timeit import default_timer as timer

class ASKFSVM:
    def __init__(self, strategy='one_vs_one', **kwargs):
        if strategy not in ['one_vs_one']:
            raise ValueError("strategy must be 'one_vs_one'")
        self.strategy = strategy
        self.kwargs = kwargs
        self.classifier = None

    # def fit(self, Ks, y):
    #     self.y = self._convert_labels(y)

    def fit(self, K, y):
        self.classes = np.unique(y)
        if len(self.classes) > 2:  # multi-class case
            self.classifier = OneVsRestClassifier(**self.kwargs)
            self.classifier.fit(K, y)
            self.time = self.classifier.diff
        else:  # binary classification case
            # TODO: check if y is equal to -1 and 1
            y_bin = np.where(y == self.classes[0], -1, 1)
            self.binary_classifier = ASKFSVMBinary(**self.kwargs)
            self.binary_classifier.fit(K, y_bin)

    def predict(self, K):
        if self.classifier is not None:  # multi-class case
            return self.classifier.predict(K)
        else:  # binary classification case
            return self.binary_classifier.predict(K)
