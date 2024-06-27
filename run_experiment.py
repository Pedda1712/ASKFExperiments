import csv
import sys
import json
import numpy as np
from voASKF import run_test_on_voASKF, ASKFvoSVM
from ASKF import run_test_on_ASKF, ASKFSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


def rbf_kernel(X1, X2, gamma=1):
    sqdist = (
        np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    return np.exp(-gamma * sqdist)


def gamma_estimate(X):
    n_features = X.shape[1]
    return 1 / (n_features * np.var(X))


def grid_search(
    classifier, Ks, labels, on_gpu, max_iter, crossv
):  # exhaustive grid search for ASKF hyperparameters
    betas = [0, -1, -10, -100]
    gammas = [0, 0.1, 1, 10, 100]
    deltas = [1, 10, 100]
    cs = [0.01, 0.1, 1, 10, 100]

    if Ks[0].shape[0] > 500: # max 500 samples for grid search
        nk = []
        i =np.arange(500)
        for k in Ks:
            nk.append(k[i, :][:, i])
        labels = labels[i].flatten()
        Ks = nk

    best_acc = 0
    best_tuple = ()
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

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_tuple = e
    return {"beta": e[0], "gamma": e[1], "delta": e[2], "C": e[3]}


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

    csv_fieldnames = [
        "# classes",
        "# samples",
        "# repetitions",
        "ASKF CPU Mean Train Accuracy",
        "ASKF CPU std-dev Train Accuracy",
        "ASKF CPU Mean Test Accuracy",
        "ASKF CPU std-dev Test Accuracy",
        "ASKF CPU Mean Execution Time",
        "ASKF CPU std-dev Execution Time",
        "ASKF GPU Mean Train Accuracy",
        "ASKF GPU std-dev Train Accuracy",
        "ASKF GPU Mean Test Accuracy",
        "ASKF GPU std-dev Test Accuracy",
        "ASKF GPU Mean Execution Time",
        "ASKF GPU std-dev Execution Time",
        "voASKF GPU Mean Train Accuracy",
        "voASKF GPU std-dev Train Accuracy",
        "voASKF GPU Mean Test Accuracy",
        "voASKF GPU std-dev Test Accuracy",
        "voASKF GPU Mean Execution Time",
        "voASKF GPU std-dev Execution Time",
    ]
    csv_lines = []
    for m in measurements:
        current_iteration += 1
        # load training and test data
        m_fname = m["data"]
        m_repeat = int(m["repeat"])

        m_f = open(m_fname, "r")
        m_json = json.load(m_f)

        m_Ks = []
        m_Ktests = []
        m_labels = []
        m_tlabels = []

        m_samples = 0
        m_classes = 0
        if m_json["type"] == "vectorial":
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
        else:
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

        measurement_line = {
            "# classes": m_classes,
            "# samples": m_samples,
            "# repetitions": m_repeat,
            "ASKF CPU Mean Train Accuracy": "-",
            "ASKF CPU std-dev Train Accuracy": "-",
            "ASKF CPU Mean Test Accuracy": "-",
            "ASKF CPU std-dev Test Accuracy": "-",
            "ASKF CPU Mean Execution Time": "-",
            "ASKF CPU std-dev Execution Time": "-",
            "ASKF GPU Mean Train Accuracy": "-",
            "ASKF GPU std-dev Train Accuracy": "-",
            "ASKF GPU Mean Test Accuracy": "-",
            "ASKF GPU std-dev Test Accuracy": "-",
            "ASKF GPU Mean Execution Time": "-",
            "ASKF GPU std-dev Execution Time": "-",
            "voASKF GPU Mean Train Accuracy": "-",
            "voASKF GPU std-dev Train Accuracy": "-",
            "voASKF GPU Mean Test Accuracy": "-",
            "voASKF GPU std-dev Test Accuracy": "-",
            "voASKF GPU Mean Execution Time": "-",
            "voASKF GPU std-dev Execution Time": "-",
        }

        sums = {
            "ASKF CPU": [],
            "ASKF CPU Train Accuracy": [],
            "ASKF CPU Test Accuracy": [],
            "ASKF GPU": [],
            "ASKF GPU Train Accuracy": [],
            "ASKF GPU Test Accuracy": [],
            "voASKF GPU": [],
            "voASKF GPU Train Accuracy": [],
            "voASKF GPU Test Accuracy": [],
        }

        print("------------START PARAMETER SEARCH--------------------")
        hypersASKF = grid_search(
            ASKFSVM, m_Ks[0], m_labels[0], on_gpu=gpu_supported, max_iter=200, crossv=5
        )
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
        print("------------END   PARAMETER SEARCH--------------------")


        for i in range(m_repeat):
            print(
                "Measurement ", current_iteration, " Repetition ", i, " of ", m_repeat
            )
            # run ASKF CPU
            # if m_samples <= max_cpu_sample_count:
            results = run_test_on_ASKF(
                m_Ks[i], m_Ktests[i], m_labels[i], m_tlabels[i], False, hypersASKF
            )
            sums["ASKF CPU Train Accuracy"].append(results["train_accuracy"])
            sums["ASKF CPU Test Accuracy"].append(results["test_accuracy"])
            sums["ASKF CPU"].append(results["time"])
            # run ASKF GPU
            if gpu_supported:
                results = run_test_on_ASKF(
                    m_Ks[i], m_Ktests[i], m_labels[i], m_tlabels[i], True, hypersASKF
                )
                sums["ASKF GPU Train Accuracy"].append(results["train_accuracy"])
                sums["ASKF GPU Test Accuracy"].append(results["test_accuracy"])
                sums["ASKF GPU"].append(results["time"])
            # run voASKF GPU
            results = run_test_on_voASKF(
                m_Ks[i], m_Ktests[i], m_labels[i], m_tlabels[i], gpu_supported, hypersVO
            )  # TODO: switch to True for the real thing
            sums["voASKF GPU Train Accuracy"].append(results["train_accuracy"])
            sums["voASKF GPU Test Accuracy"].append(results["test_accuracy"])
            sums["voASKF GPU"].append(results["time"])

        # average
        measurement_line["ASKF CPU Mean Execution Time"] = np.mean(
            np.array(sums["ASKF CPU"])
        )
        measurement_line["ASKF CPU std-dev Execution Time"] = np.std(
            np.array(sums["ASKF CPU"])
        )
        measurement_line["ASKF CPU Mean Train Accuracy"] = np.mean(
            np.array(sums["ASKF CPU Train Accuracy"])
        )
        measurement_line["ASKF CPU std-dev Train Accuracy"] = np.std(
            np.array(sums["ASKF CPU Train Accuracy"])
        )
        measurement_line["ASKF CPU Mean Test Accuracy"] = np.mean(
            np.array(sums["ASKF CPU Test Accuracy"])
        )
        measurement_line["ASKF CPU std-dev Test Accuracy"] = np.std(
            np.array(sums["ASKF CPU Test Accuracy"])
        )

        measurement_line["ASKF GPU Mean Execution Time"] = np.mean(
            np.array(sums["ASKF GPU"])
        )
        measurement_line["ASKF GPU std-dev Execution Time"] = np.std(
            np.array(sums["ASKF GPU"])
        )
        measurement_line["ASKF GPU Mean Train Accuracy"] = np.mean(
            np.array(sums["ASKF GPU Train Accuracy"])
        )
        measurement_line["ASKF GPU std-dev Train Accuracy"] = np.std(
            np.array(sums["ASKF GPU Train Accuracy"])
        )
        measurement_line["ASKF GPU Mean Test Accuracy"] = np.mean(
            np.array(sums["ASKF GPU Test Accuracy"])
        )
        measurement_line["ASKF GPU std-dev Test Accuracy"] = np.std(
            np.array(sums["ASKF GPU Test Accuracy"])
        )

        measurement_line["voASKF GPU Mean Execution Time"] = np.mean(
            np.array(sums["voASKF GPU"])
        )
        measurement_line["voASKF GPU std-dev Execution Time"] = np.std(
            np.array(sums["voASKF GPU"])
        )
        measurement_line["voASKF GPU Mean Train Accuracy"] = np.mean(
            np.array(sums["voASKF GPU Train Accuracy"])
        )
        measurement_line["voASKF GPU std-dev Train Accuracy"] = np.std(
            np.array(sums["voASKF GPU Train Accuracy"])
        )
        measurement_line["voASKF GPU Mean Test Accuracy"] = np.mean(
            np.array(sums["voASKF GPU Test Accuracy"])
        )
        measurement_line["voASKF GPU std-dev Test Accuracy"] = np.std(
            np.array(sums["voASKF GPU Test Accuracy"])
        )

        csv_lines.append(measurement_line)
        intermediate_csv = (
            csv_prefix + str(current_iteration) + "of" + str(max_iterations)
        )

        print("outputting intermediate csv: ", intermediate_csv)
        c = open(intermediate_csv + ".csv", "w", newline="")
        writer = csv.DictWriter(c, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_lines)
        c.close()

    c = open(csv_prefix + ".csv", "w", newline="")
    writer = csv.DictWriter(c, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(csv_lines)
    c.close()
