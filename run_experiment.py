import csv
import sys
import json
import numpy as np
from voASKF import run_test_on_voASKF, ASKFvoSVM
from ASKF import run_test_on_ASKF, ASKFSVM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from accumulator import Accumulator
from ordered_set import OrderedSet
from ucimlrepo import fetch_ucirepo 
import itertools

# This file runs ASKF experiments (ASKF-One-V-Rest CPU vs ASKF-One-V-Rest GPU vs voASKF GPU)
# and outputs train/test accuracy, and mean execution time + std deviation
# it outputs CSV files

import h5py
import scipy.io


# Obtain Kernel list from MAT 5.0 or MAT 7.3
def load_kernels(fname, kernel_table, label_table):
    # load mat 5
    try:
        mat = scipy.io.loadmat(fname)
        klist = []
        for k in mat[kernel_table]:
            klist.append(np.array(k[0], dtype=np.double))
        labels = mat[label_table].astype(int)
        labels = labels - np.min(labels)
        return (klist, labels)
    except:
        print("loading mat5 failed, trying mat7")
    try:
        mat = h5py.File(fname)
        kernel_list = mat[kernel_table][0]
        klist = []
        for kr in mat[kernel_table][0]:
            k = np.array(mat[kr], dtype=np.double)
            klist.append(k)
        labels = np.array(mat[label_table], dtype=int)
        labels = labels - np.min(labels)
        return (klist, labels[0])
    except:
        print("loading mat7 failed")

    raise RuntimeError("could not open mat file " + fname)


def lin_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1):
    sqdist = (
        np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    return np.exp(-gamma * sqdist)

def tanh_kernel(X1, X2, a, b):
    t = np.tanh(a * (X1 @ X2.T) + b)
    #print(np.max(t), np.min(t))
    return t

def gamma_estimate(X):
    n_features = X.shape[1]
    return 1 / (n_features * np.var(X))

# determine a,b so that tanh(a*min(x) +b) ~= -0.99 and tanh(a*max(x) +b) ~= 0.99
def tanh_parameters_estimate(X):
    max = np.max(X @ X.T)
    min = np.min(X @ X.T)
    a = (-4)/(min-max)
    b = 2 - a * max
    return a,b
    

def grid_search(
    classifier, Ks, labels, on_gpu, max_iter, crossv
):  # exhaustive grid search for ASKF hyperparameters
    betas = [-1, -10, -100]
    gammas = [0, 1, 10, 100]
    deltas = [1, 10, 100]
    cs = [0.1, 1, 10, 100]

    if Ks[0].shape[0] > 100: # max 100 samples for grid search
        nk = []
        i =np.arange(100)
        for k in Ks:
            nk.append(k[i, :][:, i])
        labels = labels[i].flatten()
        Ks = nk

    best_acc = 0
    best_tuple = ()
    total = len(list(itertools.product(betas, gammas, deltas, cs)))
    now = 1
    for e in itertools.product(betas, gammas, deltas, cs):
        A = classifier(
            max_iter=max_iter,
            on_gpu=on_gpu,
            beta=e[0],
            gamma=e[1],
            delta=e[2],
            C=e[3],
            subsample_size=1,
        )
        accuracy_test = []
        for i in range(crossv):
            _k = []
            _kval = []
            _l = None
            _ls = None
            for k in Ks:
                train_ind, test_ind, _l, _ls = train_test_split(
                    np.arange(Ks[0].shape[0]), labels, test_size=0.5, random_state=i
                )
                _k.append(k[train_ind, :][:, train_ind])
                _kval.append(k[test_ind, :][:, train_ind])
            A.fit(_k, _l)
            y_pred = A.predict(_kval)
            accuracy_test.append(accuracy_score(_ls, y_pred))

        mean_acc = np.mean(np.array(accuracy_test))
        print("grid search " + str(now) + " of " + str(total) + " " + str(mean_acc))
        now += 1
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_tuple = e
    return {"beta": best_tuple[0], "gamma": best_tuple[1], "delta": best_tuple[2], "C": best_tuple[3]}

def get_accumulators(with_cpu_ovr, on_gpu, hypersASKF, hypersVO):
    accs = []
    accs.append(Accumulator(ASKFvoSVM(max_iter=200, on_gpu=on_gpu, beta=hypersVO["beta"], gamma=hypersVO["gamma"], delta=hypersVO["delta"], C=hypersVO["C"], subsample_size=1.0),
                                "vo-askf-gpu"))
    accs.append(Accumulator(ASKFSVM(max_iter=200, subsample_size=1.0, on_gpu=on_gpu, beta=hypersASKF["beta"], gamma=hypersASKF["gamma"], delta=hypersASKF["delta"], C=hypersASKF["C"]),
                                "ovr-askf-gpu"))
    if on_gpu and with_cpu_ovr:
        accs.append(Accumulator(ASKFSVM(max_iter=200, subsample_size=1.0, on_gpu=False, beta=hypersASKF["beta"], gamma=hypersASKF["gamma"], delta=hypersASKF["delta"], C=hypersASKF["C"]),
                                "ovr-askf-cpu"))
    return accs

def get_fieldnames(csv_lines):
    s = []
    for l in csv_lines:
        s = list(OrderedSet(s) | OrderedSet(l.keys()))
    return s

args = sys.argv

if len(args) != 3:
    print("python run_experiment.py <experiment_specification.json> <use-gpu:1|0>")
    exit(1)

gpu_supported = int(args[2]) == 1

with open(args[1]) as f:
    d = json.load(f)
    csv_prefix = d["outfile_prefix"]
    measurements = d["measurements"]
    # if a dataset contains more samples than specified here, the CPU run is omitted (would take too long)
    max_cpu_sample_count = d["max_cpu_samples"]
    max_iterations = len(measurements)
    current_iteration = 0

    csv_lines = []
    for m in measurements:
        current_iteration += 1
        # load training and test data
        m_fname = m["data"]
        m_dname = m_fname
        m_repeat = int(m["repeat"])

        m_f = open(m_fname, "r")
        m_json = json.load(m_f)

        if "dataset-name" in m_json:
            m_dname = m_json["dataset-name"]

        m_Ks = []
        m_Ktests = []
        m_labels = []
        m_tlabels = []

        m_samples = 0
        m_classes = 0
        m_definiteness = 0
        m_X_train = []
        m_y_train = []
        m_X_test = []
        m_y_test = []
        if m_json["type"] == "uci":
            s = fetch_ucirepo(id=m_json["data"]["id"])
            m_X = s.data.features
            m_y = s.data.targets
            if "target" in m_json["data"]:
                m_y = m_y[m_json["data"]["target"]]
            m_X = np.array(m_X)
            _, m_y = np.unique(np.array(m_y), return_inverse=True)
            m_y = np.ndarray.flatten(m_y)
            
            m_samples = m_X.shape[0]
            m_classes = np.unique(m_y).shape[0]
            outer_psd = 0
            for i in range(m_repeat):
                _train_X, _test_X, _train_c, _test_c = train_test_split(
                    m_X, m_y, test_size=0.3, random_state=i
                )
                m_X_train.append(_train_X)
                m_X_test.append(_test_X)
                m_y_train.append(_train_c)
                m_y_test.append(_test_c)
                g_est = gamma_estimate(_train_X)
                a_est, b_est = tanh_parameters_estimate(_train_X)
                _Ks = []
                _K_test_s = []
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 0.01))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 0.1))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 10))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 100))
                _Ks.append(tanh_kernel(_train_X, _train_X, a_est, b_est))
                _Ks.append(lin_kernel(_train_X, _train_X))

                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 0.01))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 0.1))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 10))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 100))
                _K_test_s.append(tanh_kernel(_test_X, _train_X, a_est, b_est))
                _K_test_s.append(lin_kernel(_test_X, _train_X))

                psd_score = 0
                for _k in _Ks:
                    eigv, _ = np.linalg.eig(_k)
                    score = np.sum(np.abs(eigv[np.where(eigv < 0)])) / np.sum(np.abs(eigv))
                    psd_score += score
                psd_score /= len(_Ks)
                outer_psd += psd_score
                
                m_Ks.append(_Ks)
                m_Ktests.append(_K_test_s)
                m_labels.append(_train_c)
                m_tlabels.append(_test_c)
            m_definiteness = outer_psd / m_repeat
            print(m_definiteness)
        elif m_json["type"] == "vectorial":
            m_X = m_json["data"]["x"]
            m_c = m_json["data"]["c"]
            m_X = np.array(m_X).astype(float).T
            m_samples = m_X.shape[0]
            m_c = np.array(m_c).astype(int)
            m_classes = np.unique(m_c).shape[0]
            for i in range(m_repeat):
                _train_X, _test_X, _train_c, _test_c = train_test_split(
                    m_X, m_c, test_size=0.3, random_state=i
                )
                g_est = gamma_estimate(_train_X)
                _Ks = []
                _K_test_s = []
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 0.01))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 0.1))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 10))
                _Ks.append(rbf_kernel(_train_X, _train_X, g_est * 100))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 0.01))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 0.1))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 10))
                _K_test_s.append(rbf_kernel(_test_X, _train_X, g_est * 100))

                m_Ks.append(_Ks)
                m_Ktests.append(_K_test_s)
                m_labels.append(_train_c)
                m_tlabels.append(_test_c)
        elif m_json["type"] == "kernels":
            whole_ks = []
            whole_c = []
            try:
                whole_ks, whole_c = load_kernels(m_json["data"]["file"], m_json["data"]["kernels"], m_json["data"]["labels"])
            except:
                print("failed loading set ", m_json["data"]["file"], " skipping")
                continue
            m_samples = len(whole_ks[0])
            m_c = np.array(whole_c)
            m_classes = np.unique(np.array(whole_c)).shape[0]
            for i in range(m_repeat):
                train_ind, test_ind, c_train, c_test = train_test_split(
                    np.arange(m_samples), m_c, test_size=0.3, random_state=i
                )

                _Ks = []
                _K_test_s = []
                for k in whole_ks:
                    nk = np.array(k)
                    _train = nk[train_ind, :][:, train_ind]
                    _test = nk[test_ind, :][:, train_ind]
                    _Ks.append(_train)
                    _K_test_s.append(_test)
                m_Ks.append(_Ks)
                m_Ktests.append(_K_test_s)
                m_labels.append(c_train)
                m_tlabels.append(c_test)
        print("grid search OVR")
        #hypersASKF = {"beta": -10, "gamma": 50, "delta": 10, "C": 10}
        #hypersVO = {"beta": -10, "gamma": 50, "delta": 10, "C": 10}
        hypersASKF = grid_search(
            ASKFSVM, m_Ks[0], m_labels[0], on_gpu=gpu_supported, max_iter=200, crossv=5
        )
        print("grid search VO")
        hypersVO = grid_search(
            ASKFvoSVM,
            m_Ks[0],
            m_labels[0],
            on_gpu=gpu_supported,
            max_iter=200,
            crossv=5,
        )
        print("hyperparameters OVR ", hypersASKF)
        print("hyperparameters VO", hypersVO)


        cpu = m_samples <= max_cpu_sample_count
        accs = get_accumulators(cpu, gpu_supported, hypersASKF, hypersVO)
        
        estimator_KNN = KNeighborsClassifier(algorithm='auto')
        parameters_KNN = {
            'n_neighbors': (1,5, 10),
            'weights': ['distance']
            }           
        grid_search_KNN = GridSearchCV(
            estimator=estimator_KNN,
            param_grid=parameters_KNN,
            scoring = 'accuracy',
            n_jobs = -1,
            cv = 5
        )
        knn = Accumulator(grid_search_KNN, "kNN")
        for i in range(m_repeat):
            print("measurement " + str(i+1) + "/" + str(m_repeat) + " of experiment " + str(current_iteration))
            for a in accs:
                a.run(m_Ks[i], m_Ktests[i], m_labels[i], m_tlabels[i])
            if len(m_X_train) > 0: # vectorial uci data -> do kNN as comparision
                knn.run(m_X_train[i], m_X_test[i], m_labels[i], m_tlabels[i])

        measurement_line = {
            "dataset-name": m_dname,
            "n-samples": m_samples,
            "n-classes": m_classes,
            "n-cv": m_repeat,
            "psd-score": m_definiteness
        }
        for a in accs:
            measurement_line = measurement_line | a.result()
        if len(m_X_train) > 0:
            measurement_line = measurement_line | knn.result()
        # average
        csv_lines.append(measurement_line)
        intermediate_csv = (
            csv_prefix + str(current_iteration) + "of" + str(max_iterations)
        )

        print("outputting intermediate csv: ", intermediate_csv)
        c = open(intermediate_csv + ".csv", "w", newline="")
        writer = csv.DictWriter(c, fieldnames=get_fieldnames(csv_lines))
        writer.writeheader()
        writer.writerows(csv_lines)
        c.close()

    c = open(csv_prefix + ".csv", "w", newline="")
    writer = csv.DictWriter(c, fieldnames=get_fieldnames(csv_lines))
    writer.writeheader()
    writer.writerows(csv_lines)
    c.close()
