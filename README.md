# Setup
Run to setup python, python env and datasets
    
'bash scripts/setup_env.sh'

# Running program
Firstly run *python -m src.data_preprocess.nist_dataset.py* to clean and concat the dataset

Secondly run *python -m src.main --iterations \<number\>* to train the model and evaluate
