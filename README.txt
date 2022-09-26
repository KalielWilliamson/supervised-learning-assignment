## Instructions for running this code
1. Clone the repository
`git clone https://github.com/KalielWilliamson/supervised-learning-assignment.git`

2. Download the all_train hepmass dataset from [here](https://raw.githubusercontent.com/jeffheaton/jheaton-ds2/main/kdd-with-columns.csv), rename it to 'hepmass.csv', and place it in the `data` folder.

3. Create a new conda environment with the required packages:
`conda env create -f environment.yml -n <env_name>`

4. Activate the environment:
`conda activate <env_name>`

5. Run the exploratory data analysis notebook:
`jupyter notebook EDA.ipynb`

6. Run the hyperparameter tuning script:
`python tuner.py`

7. In another terminal tab/window, activate the conda environment, and visualize the running results of the hyperparameter tuning script:
`optuna-dashboard sqlite:///db.sqlite3`

8. Run the tuner_analysis notebook to analyze the results of the hyperparameter tuning script:
`jupyter notebook tuner_analysis.ipynb`
