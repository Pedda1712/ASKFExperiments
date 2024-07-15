import numpy as np
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# class that accumulates CV results for a model

class Accumulator():

    # model is an already instantiated object (with hyperparameters already set)
    def __init__(self, model, model_id):
        self.model = model
        self.model_id = model_id
        self.runs = 0
        self.quantities = {
            "test-accuracy": [],
            "train-accuracy": [],
            "train-time": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "sv-count": []
        }

    def clear(self):
        self.runs = 0
        self.quantities = {
            "test-accuracy": [],
            "train-accuracy": [],
            "train-time": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "sv-count": []
        }

    def run(self, Ks, Ktests, labels, testlabels):
        start = timer()
        self.model.fit(Ks, labels)
        time = timer() - start
        tlabels = self.model.predict(Ks)

        plabels = self.model.predict(Ktests)

        self.quantities["test-accuracy"].append(accuracy_score(testlabels, plabels))
        print("acc ", self.model_id, " ", self.quantities["test-accuracy"])
        self.quantities["train-accuracy"].append(accuracy_score(labels, tlabels))
        self.quantities["train-time"].append(time)
        self.quantities["precision"].append(precision_score(testlabels, plabels, average="weighted", zero_division=0.0))
        self.quantities["recall"].append(recall_score(testlabels, plabels, average="weighted"))
        self.quantities["f1"].append(f1_score(testlabels, plabels, average="weighted"))
        self.quantities["sv-count"].append(self.model.getSVCount())
    def result(self):
        out = {}
        for key, value in self.quantities.items():
            mean_name = self.model_id + "-" + key + "-mean"
            std_name = self.model_id + "-" + key + "-std"
            out[mean_name] = np.mean(np.array(self.quantities[key]))
            out[std_name] = np.std(np.array(self.quantities[key]))
        return out
