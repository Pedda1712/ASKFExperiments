import numpy as np

class kNN:

    def __init__(self, k):
        self.k = k

    def fit(self, Ks, labels):
        self.labels = labels
        return # we get the kernels in the predict method eitherway

    def predict(self, Ks):
        K = np.zeros(Ks[0].shape)
        for _k in Ks:
            mini = np.min(_k)
            offset = 0
            if mini < 0:
                offset = -mini
            K = K + (_k + offset)
        idx = K.argsort(axis=1)[:,::-1][:,:self.k]
        picker = np.vectorize(lambda i: self.labels[i])
        lbls = picker(idx)
        vals, counts = np.unique(lbls, axis=1, return_counts=True)
        u, indices = np.unique(lbls, return_inverse=True)
        axis = 1
        res = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(lbls.shape),
                                              None, np.max(indices) + 1), axis=axis)]
        return res
