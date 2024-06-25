import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from .models.askfsvm import ASKFSVM
from timeit import default_timer as timer

def run_test_on_ASKF(Ks, Ktests, labels, testlabels, use_gpu):

    svm = ASKFSVM(max_iter=200, subsample_size=1.0, on_gpu=use_gpu)
    # Train the SVM
    start = timer()
    svm.fit(Ks, labels)
    time = timer() - start
    
    # Use the trained SVM to make predictions on the test set
    y_pred = svm.predict(Ktests)
    y_train_pred = svm.predict(Ks)

    # Compute the accuracy of the predictions
    accuracy_test = accuracy_score(testlabels, y_pred)
    accuracy_train = accuracy_score(labels, y_train_pred)

    print("ASKF run took ", time)
    results = {
        "time": time,
        "test_accuracy": accuracy_test,
        "train_accuracy": accuracy_train
    }
    print(results)
    return results

