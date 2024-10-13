# Setup
Run to setup python, python env and datasets
    
'bash scripts/setup_env.sh'

# Running program
Firstly run *python -m src.data_preprocess.nist_dataset.py* to clean and concat the dataset

Secondly run *python -m src.main* to train the model and evaluate

# SPECIFIC FOR THIS BRANCH
No setup for getting the CSVs is currently setup, so a data_root folder must be created in the ukdata folder with the 5 first files from the zip. 
This will be changed.
