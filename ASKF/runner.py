import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from .models.askfsvm import ASKFSVM
from timeit import default_timer as timer


# Define a function to generate a linear kernel
def rbf_kernel(X1, X2, gamma=1.0):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sqdist)

def run_test_on_ASKF(X_train, y_train, X_test, y_test, use_gpu):
    K_train = rbf_kernel(X_train, X_train, 0.01)
    K_test = rbf_kernel(X_test, X_train, 0.01)
    
    K0_train = rbf_kernel(X_train, X_train, 0.1)
    K0_test = rbf_kernel(X_test,X_train,   0.1)

    K1_train = rbf_kernel(X_train, X_train, 1.0)
    K1_test = rbf_kernel(X_test,X_train,   1.0)

    K2_train = rbf_kernel(X_train, X_train, 10.0)
    K2_test = rbf_kernel(X_test,X_train,   10.0)

    K3_train = rbf_kernel(X_train, X_train, 100.0)
    K3_test = rbf_kernel(X_test,X_train,   100.0)

    K_train = [K_train, K0_train, K1_train, K2_train, K3_train]
    K_test = [K_test, K0_test, K1_test, K2_test, K3_test]

    # Initialize the SVM with maximum iterations and subsample size
    svm = ASKFSVM(max_iter=200, subsample_size=1.0, on_gpu=use_gpu)
    # Train the SVM
    start = timer()
    svm.fit(K_train, y_train)
    time = timer() - start
    
    # Use the trained SVM to make predictions on the test set
    y_pred = svm.predict(K_test)
    y_train_pred = svm.predict(K_train)

    # Compute the accuracy of the predictions
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)

    print("ASKF run took ", time)
    results = {
        "time": time,
        "test_accuracy": accuracy_test,
        "train_accuracy": accuracy_train
    }
    return results

