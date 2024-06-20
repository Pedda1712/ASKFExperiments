import numpy as np
from .Model import ASKFvoSVM
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

def run_test_on_voASKF(Ks, Ktests, labels, testlabels, use_gpu):
        start = timer()
        model = ASKFvoSVM(Ks, labels, max_iter=200, on_gpu=use_gpu)
        time = timer() - start
        tlabels = model.predict(Ks)

        # model test accuracy
        plabels = model.predict(Ktests)

        print("voASKF run took", time)
        results = {
                "time": time,
                "train_accuracy": accuracy_score(labels, tlabels),
                "test_accuracy": accuracy_score(testlabels, plabels)
        }
        return results

