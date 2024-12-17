# Setup
To setup the environment on Ubuntu run:
```bash
sudo ./scripts/setup_env.sh
```

# Cleaning the Datasets
Run the following commands to clean datasets:
```bash
python -m src.data_preprocess.nist.nist_dataset
python -m src.data_preprocess.ukdata.ukdata_dataset
```

# Runing the Experiments
1. Firstly you will have to run the topmost code block to initialize the Jupyter environment with imports and constants (This step every time a change occurs in the environment)
2. Secondly you have to run the second code block to train and save the model for the experiment. (This step is once for every experiment)
3. Lastly you can run the last code block which is used for evaluation, after training a model it will be saved and you can re-evaluate as long as the save file exists. (This step at your leisure)

# Hypertuning
If you wish to do hyperparameter tuning this can be done by running:
```bash
python -m src.hyper_tuning
```
But you will your self have to change the hyperparameters in import under util in experiments or in the file you are running.

# Tests
Tests can be run with python run_tests.py
