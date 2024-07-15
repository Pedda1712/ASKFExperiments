from collections import Counter
import numpy as np
from ..models.askfsvm_binary import ASKFSVMBinary
from timeit import default_timer as timer

class OneVsRestClassifier:
    def __init__(self, **kwargs):
        self.models = {}
        self.class_map = {}
        self.kwargs = kwargs
        self.max_iter = kwargs['max_iter']
        self.subsample_size = kwargs['subsample_size']
        self.mp = True if "mp" in kwargs.keys() and kwargs['mp'] else False
        self.classes = None

    def fit_single(self, K, y):
        for idx, i in enumerate(self.classes):
            y_binary = np.where(y == i, 1, -1)
            model = ASKFSVMBinary(**self.kwargs)
            model.fit(K, y_binary)
            self.models[i] = model
            self.class_map[idx + 1] = i  # 1 maps to first class, -1 maps to rest

    def getSVCount(self):
        svinds = np.array([])
        for class_i, model in self.models.items():
            svinds = np.append(svinds, model.svinds)
        return np.unique(svinds).shape[0]
#    def fit_multi(self, K, y):
#        # init ray
#        ray.init()
#        # push K to shared object storage
#        data_id = ray.put(K)
#
#        rays = list()
#        for idx, i in enumerate(self.classes):
#            rays.append(fit_.remote(data_id, y, idx, i, self.max_iter, self.subsample_size))
#        # wait until all ray jobs finished
#        res = ray.get(rays)
#
#        # fill the class variables
#        for entry in res:
#            self.models[entry[0]] = entry[2]
#            self.class_map[entry[1] + 1] = i  # 1 maps to first class, -1 maps to rest
#
#        # shutdown ray
#        ray.shutdown()
#
    def fit(self, K, y):
        start = timer()
        self.classes = np.unique(y)
        self.fit_single(K, y)
        stop = timer()
        self.diff = stop - start
        #print("OneVsRestClassifier fit took " + str(self.diff))


    def predict(self, K):
        predictions = []
        max_classi = 0
        for class_i, _ in self.models.items():
            if class_i > max_classi:
                max_classi = class_i
        score_matrix = np.zeros(shape=(len(K[0]),max_classi+1))
        for class_i, model in self.models.items():
            scores = model.decision_function(K)
            score_matrix[:, class_i] = scores
        predictions = np.argmax(score_matrix, axis=1)

        return predictions
