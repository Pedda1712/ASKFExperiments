from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import sys
import json

args = sys.argv

if len(args) != 4:
    print(args)
    print("python gen.py <classes> <features> <samples>")
    exit()

n_classes = int(args[1])
n_dimensions = int(args[2])
n_samples = int(args[3])

X, labels = make_blobs(n_samples=n_samples,
                       centers=n_classes,
                       n_features=n_dimensions,
                       random_state=0)

dict = {"type": "vectorial", "data": {}}

train_features = []
for i in range(0, n_dimensions):
    x_train = X[:, i]
    train_features.append(x_train.tolist())

dict["data"]["x"] = train_features
dict["data"]["c"] = labels.tolist()

json_object = json.dumps(dict, indent=4)
json_file = "data_" + str(n_classes) + "_" + str(n_dimensions) + "_" + str(n_samples) + ".json"

# Writing to sample.json
with open(json_file, "w") as outfile:
    outfile.write(json_object)
