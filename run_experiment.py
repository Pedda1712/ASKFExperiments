import csv
import sys
import json
import numpy as np
from voASKF import run_test_on_voASKF
from ASKF import run_test_on_ASKF

# This file runs ASKF experiments (ASKF-One-V-Rest CPU vs ASKF-One-V-Rest GPU vs voASKF GPU)
# and outputs train/test accuracy, and mean execution time + std deviation
# it outputs CSV files

args = sys.argv

if len(args) != 2:
    print("python run_experiment.py <experiment_specification.json>")
    exit(1)

# if a dataset contains more samples than specified here, the CPU run is omitted (would take too long)
max_cpu_sample_count = 800

with open(args[1]) as f:
    d = json.load(f)
    csv_prefix = d["outfile_prefix"]
    measurements = d["measurements"]
    max_iterations = len(measurements)
    current_iteration = 0

    csv_fieldnames = ["# classes",
                      "# samples",
                      "# repetitions",
                      "ASKF CPU Train Accuracy",
                      "ASKF CPU Test Accuracy",
                      "ASKF CPU Mean Execution Time",
                      "ASKF CPU std-dev Execution Time",
                      "ASKF GPU Train Accuracy",
                      "ASKF GPU Test Accuracy",
                      "ASKF GPU Mean Execution Time",
                      "ASKF GPU std-dev Execution Time",
                      "voASKF GPU Train Accuracy",
                      "voASKF GPU Test Accuracy",
                      "voASKF GPU Mean Execution Time",
                      "voASKF GPU std-dev Execution Time"
                      ]
    csv_lines = []    
    for m in measurements:
        current_iteration += 1
        # load training and test data
        m_fname = m["data"]
        m_repeat = int(m["repeat"])

        m_f = open(m_fname, "r")
        m_json = json.load(m_f)

        m_train_X = m_json["train"]["x"]
        m_train_c = m_json["train"]["c"]
        m_test_X = m_json["test"]["x"]
        m_test_c = m_json["test"]["c"]

        m_train_X = np.array(m_train_X).astype(float).T
        m_train_c = np.array(m_train_c).astype(int)
        m_test_X = np.array(m_test_X).astype(float).T
        m_test_c = np.array(m_test_c).astype(int)

        m_samples = m_train_X.shape[0] + m_test_X.shape[0]
        measurement_line = {
            "# classes": np.unique(m_train_c).shape[0],
            "# samples": m_samples,
            "# repetitions": m_repeat,
            "ASKF CPU Train Accuracy": "-",
            "ASKF CPU Test Accuracy": "-",
            "ASKF CPU Mean Execution Time": "-",
            "ASKF CPU std-dev Execution Time": "-",
            "ASKF GPU Train Accuracy": "-",
            "ASKF GPU Test Accuracy": "-",
            "ASKF GPU Mean Execution Time": "-",
            "ASKF GPU std-dev Execution Time": "-",
            "voASKF GPU Train Accuracy": "-",
            "voASKF GPU Test Accuracy": "-",
            "voASKF GPU Mean Execution Time": "-",
            "voASKF GPU std-dev Execution Time": "-"
        }

        time_sum = {
            "ASKF CPU": [],
            "ASKF GPU": [],
            "voASKF GPU": []
        }

        for i in range(m_repeat):
            print("Measurement ", current_iteration, " Repetition ", i, " of ", m_repeat)
            # run ASKF CPU
            if m_samples <= max_cpu_sample_count:
                results = run_test_on_ASKF(m_train_X, m_train_c, m_test_X, m_test_c, False)
                measurement_line["ASKF CPU Train Accuracy"] = results["train_accuracy"]
                measurement_line["ASKF CPU Test Accuracy"] = results["test_accuracy"]
                time_sum["ASKF CPU"].append(results["time"])
            # run ASKF GPU
            results = run_test_on_ASKF(m_train_X, m_train_c, m_test_X, m_test_c, True)
            measurement_line["ASKF GPU Train Accuracy"] = results["train_accuracy"]
            measurement_line["ASKF GPU Test Accuracy"] = results["test_accuracy"]
            time_sum["ASKF GPU"].append(results["time"])
            # run voASKF GPU
            results = run_test_on_voASKF(m_train_X, m_train_c, m_test_X, m_test_c, True) # TODO: switch to True for the real thing
            measurement_line["voASKF GPU Train Accuracy"] = results["train_accuracy"]
            measurement_line["voASKF GPU Test Accuracy"] = results["test_accuracy"]
            time_sum["voASKF GPU"].append(results["time"])

        # average
        measurement_line["ASKF CPU Mean Execution Time"] = np.mean(np.array(time_sum["ASKF CPU"]))
        measurement_line["ASKF CPU std-dev Execution Time"] = np.std(np.array(time_sum["ASKF CPU"]))
        measurement_line["ASKF GPU Mean Execution Time"] = np.mean(np.array(time_sum["ASKF GPU"]))
        measurement_line["ASKF GPU std-dev Execution Time"] = np.std(np.array(time_sum["ASKF GPU"]))
        measurement_line["voASKF GPU Mean Execution Time"] = np.mean(np.array(time_sum["voASKF GPU"]))
        measurement_line["voASKF GPU std-dev Execution Time"] = np.std(np.array(time_sum["voASKF GPU"]))

        csv_lines.append(measurement_line)
        intermediate_csv = csv_prefix + str(current_iteration) + "of" + str(max_iterations)

        print("outputting intermediate csv: ", intermediate_csv)
        c = open(intermediate_csv + ".csv", 'w', newline='')
        writer = csv.DictWriter(c, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_lines)
        c.close()

    c = open(csv_prefix + ".csv", 'w', newline='')
    writer = csv.DictWriter(c, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(csv_lines)
    c.close()
