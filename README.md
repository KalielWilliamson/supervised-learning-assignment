## Instructions for running this code

1. Download the all_train hepmass dataset from [here](https://raw.githubusercontent.com/jeffheaton/jheaton-ds2/main/kdd-with-columns.csv), rename it to 'hepmass.csv', and place it in the `data` folder.

2. Create a new conda environment with the required packages:
`conda env create -f environment.yml -n <env_name>`

3. Activate the environment:
`conda activate <env_name>`

4. Run the exploratory data analysis notebook:
`jupyter notebook EDA.ipynb`

5. Run the hyperparameter tuning script:
`python tuner.py`

6. In another terminal tab/window, activate the conda environment, and visualize the running results of the hyperparameter tuning script:
`optuna-dashboard sqlite:///db.sqlite3`

7. Run the tuner_analysis notebook to analyze the results of the hyperparameter tuning script:
`jupyter notebook tuner_analysis.ipynb`
