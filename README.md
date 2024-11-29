# Setup
Run to setup python, python env and datasets
    
'bash scripts/setup_env.sh'

# Running program
Firstly run *python -m src.data_preprocess.nist.nist_dataset* & *python -m src.data_preprocess.ukdata.ukdata_dataset* 

Secondly you can run the experiments in the Experiments folder.
    -   Firstly you will have to run the top most code block in order to initialze the jupyter envirement with imports and constants (This step every time a change ocours in the enviroment)
    -   Secondly you have to run the second code block to train and save the model for the experiment. (This step once for every experiment)
    -   Lastly you can run the last code block which is used for evaluation, after training of a model it will be saved and you can re evaluate as long as the save file exist. (This step at your leisure)

In order to try things in non jupyter enviroment main can be run with *python -m src.main* or *python -m src.main - d* for debug

If you wish to do hyperparameter tuning this can be done by running *python -m src.hyper_tuning*, but you wil your self have to change the hyper parameters in import under util in experiments or in the file you are running.

# Tests
Tests can be run with python run_tests.py
