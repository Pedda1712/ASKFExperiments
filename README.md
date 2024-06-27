# ASKF Experiments
Automated script that runs hyperparameter search and then crossvalidation on datasets with the ASKF OvR and voASKF adaptive learners. 

## Usage
Create and setup venv:
```bash
python3 -m venv .
. bin/activate
pip install -r requirements.txt
```

Run an experiment WITHOUT GPU:
```bash
. bin/activate
python run_experiment.py cluster_experiment.json 0
```
Run an experiment WITH GPU (requires you to install cupy-11 before):
```bash
. bin/activate
python run_experiment.py cluster_experiment.json 1
```

Results after each dataset is finished will be output as csv files.

## Real Data
To run the cluster experiment, drop the mat files into `data/cluster/`. Your .mat files should have the same name as the json files already there.
Commands for starting the experiment on real data can be found above.
