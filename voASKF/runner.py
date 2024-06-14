import numpy as np
from .Model import ASKFvoSVM
from sklearn.metrics import accuracy_score

def rbf_kernel(X1, X2, gamma=1):
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sqdist)

def run_test_on_voASKF(X, labels, X_test, y_test, use_gpu):
        # train model on 5 gaussian kernels
        K1 = rbf_kernel(X, X, 0.01)
        K2 = rbf_kernel(X, X, 0.1)
        K3 = rbf_kernel(X, X, 1)
        K4 = rbf_kernel(X, X, 10)
        K5 = rbf_kernel(X, X, 100)
        model = ASKFvoSVM([K1, K2, K3, K4, K5], labels, max_iter=200, on_gpu=use_gpu)
        tlabels = model.predict([K1, K2, K3, K4, K5])

        # model test accuracy
        K1_test = rbf_kernel(X_test, X, 0.01)
        K2_test = rbf_kernel(X_test, X, 0.1)
        K3_test = rbf_kernel(X_test, X, 1)
        K4_test = rbf_kernel(X_test, X, 10)
        K5_test = rbf_kernel(X_test, X, 100)
        plabels = model.predict([K1_test, K2_test, K3_test, K4_test, K5_test])

        results = {
                "time": model.time,
                "train_accuracy": accuracy_score(labels, tlabels),
                "test_accuracy": accuracy_score(y_test, plabels)
        }
        return results

