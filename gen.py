from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import sys
import json

args = sys.argv

if len(args) != 5:
    print(args)
    print("python gen.py <classes> <features> <samples> <test-ratio>")
    exit()

n_classes = int(args[1])
n_dimensions = int(args[2])
n_samples = int(args[3])
test_split = float(args[4])

X, labels = make_blobs(n_samples=n_samples,
                       centers=n_classes,
                       n_features=n_dimensions,
                       random_state=0)

X_train, X_test, c_train, c_test = train_test_split(X,
                                                    labels,
                                                    test_size=test_split,
                                                    random_state=0)

dict = {
    "train": {},
    "test": {}
}

train_features = []
test_features = []
for i in range(0, n_dimensions):
    x_train = X_train[:, i]
    x_test = X_test[:, i]
    train_features.append(x_train.tolist())
    test_features.append(x_test.tolist())

dict["train"]["x"] = train_features
dict["test"]["x"] = test_features
dict["train"]["c"] = c_train.tolist()
dict["test"]["c"] = c_test.tolist()

json_object = json.dumps(dict, indent=4)
json_file = "data_" + str(n_classes) + "_" + str(n_dimensions) + "_" + str(n_samples) + "_" + str(test_split) + ".json"

# Writing to sample.json
with open(json_file, "w") as outfile:
    outfile.write(json_object)
