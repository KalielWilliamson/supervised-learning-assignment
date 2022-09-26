import os
import warnings
from datetime import datetime

import click
import optuna
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from optuna._callbacks import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

random_state = 42

global x_train
global y_train
global x_val
global y_val
global x_test
global y_test
global n_rows
n_rows = 100000


def train_test_validation(df, feature_cols, target_cols, train, test, validation):
    assert train + test + validation == 1, "Data splits must equal 1"

    x_train_val, x_test, y_train_val, y_test = train_test_split(df[feature_cols],
                                                                df[target_cols],
                                                                test_size=test,
                                                                random_state=random_state)
    if validation > 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                          y_train_val,
                                                          test_size=test,
                                                          random_state=random_state)
    else:
        x_train, x_val, y_train, y_val = x_train_val, pd.DataFrame(), y_train_val, pd.DataFrame()

    return x_train.astype(float), y_train.astype(float), x_val.astype(float), y_val.astype(float), x_test.astype(
        float), y_test.astype(float)


def encode_text_dummy(df, name) -> pd.DataFrame:
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    return df.drop(name, axis=1, inplace=True)


def download_kddcup99_dataset() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # Load dataset
    from sklearn.datasets import fetch_kddcup99
    data = fetch_kddcup99(shuffle=True, random_state=random_state, data_home="./data/kddcup99", as_frame=True)
    df = data.frame.dropna().sample(n=n_rows, random_state=random_state)
    categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    enc = preprocessing.OrdinalEncoder()
    le = preprocessing.LabelEncoder()

    df[categorical_features] = enc.fit_transform(df[categorical_features])

    df[data.feature_names] = Normalizer().fit_transform(X=df[data.feature_names].fillna(0))
    df[data.target_names] = le.fit_transform(df[data.target_names].values).reshape(-1, 1)

    return train_test_validation(df, data.feature_names, data.target_names, 0.8, 0.1, 0.1)


def download_hepmass_dataset() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = pd.read_csv("data/hepmass.csv").sample(n=n_rows, random_state=random_state)
    df = df.dropna()
    df.rename(columns={'# label': 'labels'}, inplace=True)
    df.dropna(inplace=True, axis=1)
    feature_columns = df.drop('labels', axis=1).columns
    df[feature_columns] = Normalizer().fit_transform(df[feature_columns])

    return train_test_validation(df, feature_columns, ['labels'], 0.8, 0.1, 0.1)


def study(objective, study_name, n_trials, storage):
    if 'artificial_neural_network' in objective.__name__:
        # Artificial Neural Network case
        # Because random search is only used, the direction of the optimization problem does not cause any problems
        # such as data leakage on the test set.
        # Doing things this way was simplest and easiest to implement.
        directions = ['minimize', 'minimize', 'minimize']
    else:
        directions = ['minimize', 'minimize']
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    experiment = optuna.create_study(study_name=study_name,
                                     directions=directions,
                                     sampler=optuna.samplers.RandomSampler(),
                                     load_if_exists=True,
                                     storage=storage)  # Create a new study.
    experiment.optimize(objective,
                        n_trials=n_trials,
                        callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
                        show_progress_bar=True)
    return experiment


def get_cross_validation_score(classifier, x, y) -> float:
    scores = cross_validate(
        estimator=classifier,
        X=x,
        y=y,
        cv=5,
        scoring='neg_mean_squared_error',
    )

    return -1 * scores['test_score'].mean()


def evaluate_classifier(classifier, cross_validate=False) -> (float, float):
    classifier.fit(x_train, y_train)
    if cross_validate:
        train_error = get_cross_validation_score(classifier, x_train, y_train)
    else:
        train_error = mean_squared_error(y_train, classifier.predict(x_train))
    validation_error = mean_squared_error(y_val, classifier.predict(x_val))

    print(f"\nTrain error: {train_error}")
    print(f"\nValidation error: {validation_error}")

    return train_error, validation_error


def get_dummy_y(y):
    dummy_y = pd.get_dummies(y.iloc[0].astype(str))
    dummy_y.columns = [f"y_{i}" for i in range(0, len(dummy_y.columns))]
    return dummy_y


def artificial_neural_networks_study(trial) -> (float, float, float):
    param_grid = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        'epochs': 1 #trial.suggest_int("epochs", 10, 100, log=True)
    }

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=param_grid['learning_rate'], nesterov=False, name="SGD"
    )
    dummies = pd.get_dummies(pd.concat([y_train, y_val, y_test])['labels'])

    y_train_dummy = dummies.iloc[:len(y_train)]
    y_val_dummy = dummies.iloc[len(y_train):len(y_train) + len(y_val)]
    y_test_dummy = dummies.iloc[len(y_train) + len(y_val):]

    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1], activation='tanh'))
    model.add(Dense(50, input_dim=x_train.shape[1], activation='tanh'))
    model.add(Dense(10, input_dim=x_train.shape[1], activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(y_train_dummy.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(x_train, y_train_dummy, verbose=1, epochs=param_grid['epochs'])

    train_error = model.evaluate(x_train, y_train_dummy)[1]
    validation_error = model.evaluate(x_val, y_val_dummy)[1]
    test_error = model.evaluate(x_test, y_test_dummy)[1]

    return train_error, validation_error, test_error


def artificial_neural_networks_binary_study(trial) -> (float, float, float):
    param_grid = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        'epochs': trial.suggest_int("epochs", 10, 100, log=True)
    }

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=param_grid['learning_rate'], nesterov=False, name="SGD"
    )

    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1], activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=param_grid['epochs'])

    train_error = model.evaluate(x_train, y_train)[1]
    validation_error = model.evaluate(x_val, y_val)[1]
    test_error = model.evaluate(x_test, y_test)[1]

    return train_error, validation_error, test_error


def ada_boost_decision_tree_study(trial) -> (float, float):
    param_grid = {
        'alpha_pruning': trial.suggest_float('alpha_pruning', 1e-10, 1e2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 2, 10, log=True)
    }

    classifier = AdaBoostClassifier(
        tree.DecisionTreeClassifier(random_state=random_state,
                                    criterion='entropy',
                                    ccp_alpha=param_grid['alpha_pruning']),
        n_estimators=param_grid['n_estimators'],
        random_state=random_state
    )
    return evaluate_classifier(classifier, cross_validate=False)


def decision_tree_study(trial) -> (float, float):
    param_grid = {
        'alpha_pruning': trial.suggest_float('alpha_pruning', 1e-10, 1e2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 100, log=True)
    }

    classifier = tree.DecisionTreeClassifier(random_state=random_state,
                                             criterion='entropy',
                                             ccp_alpha=param_grid['alpha_pruning'],
                                             max_depth=param_grid['max_depth'])
    return evaluate_classifier(classifier, cross_validate=True)


def k_nearest_neighbors_study(trial) -> (float, float):
    param_grid = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 100, log=True),
        'minikowski_power': trial.suggest_categorical('minikowski_power', [1, 2]),
        'distance_metric': trial.suggest_categorical('distance_metric', ['uniform', 'distance'])
    }

    classifier = KNeighborsClassifier(n_neighbors=param_grid['n_neighbors'],
                                      weights=param_grid['distance_metric'],
                                      algorithm='kd_tree',
                                      p=param_grid['minikowski_power'])

    return evaluate_classifier(classifier, cross_validate=True)


def support_vector_machine_study(trial) -> (float, float):
    param_grid = {
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'C': trial.suggest_float("C", 1e-10, 1e2, log=True),
    }
    classifier = SVC(kernel=param_grid['kernel'], C=param_grid['C'], random_state=random_state)

    return evaluate_classifier(classifier, cross_validate=True)


@click.command()
@click.option('--n_trials', '-n', help='Number of trials to run', default=20, type=int)
@click.option('--study_tag', '-t', help='The id of studies to run',
              default=datetime.now().strftime('%Y:%m:%d:%H:%M:%S'), type=str)
@click.option('--storage', '-s', help='The storage location for the studies', default="sqlite:///db.sqlite3", type=str)
@click.option('--n_row_used', '-nr', help='The number of rows used in this study', default=100000, type=int)
@click.option('--mode', '-m', help='The mode to run the studies in', default='all',
              type=click.Choice(['all', 'single']))
@click.option('--dataset', '-d', help='The name of the experiment dataset to run', default='kddcup99', type=str)
def main(n_trials, study_tag, storage, n_row_used, mode, dataset):
    global x_train
    global y_train
    global x_val
    global y_val
    global x_test
    global y_test
    global n_rows

    n_rows = n_row_used

    if mode == 'all':
        dataset = 'kddcup99'
        study_id = f"{dataset}:{study_tag}"
        x_train, y_train, x_val, y_val, x_test, y_test = download_kddcup99_dataset()
        # study(ada_boost_decision_tree_study, f'Ada Boost Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(artificial_neural_networks_study, f'Artificial Neural Networks {study_id}', n_trials=n_trials,
              storage=storage)
        study(decision_tree_study, f'Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(k_nearest_neighbors_study, f'K Nearest Neighbors {study_id}', n_trials=n_trials, storage=storage)
        study(support_vector_machine_study, f'Support Vector Machine {study_id}', n_trials=n_trials, storage=storage)

        dataset = 'hepmass'
        study_id = f"{dataset}:{study_tag}"
        x_train, y_train, x_val, y_val, x_test, y_test = download_hepmass_dataset()
        study(ada_boost_decision_tree_study, f'Ada Boost Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(artificial_neural_networks_binary_study, f'Artificial Neural Networks {study_id}', n_trials=n_trials, storage=storage)
        study(decision_tree_study, f'Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(k_nearest_neighbors_study, f'K Nearest Neighbors {study_id}', n_trials=n_trials, storage=storage)
        study(support_vector_machine_study, f'Support Vector Machine {study_id}', n_trials=n_trials, storage=storage)

    elif mode == 'single':
        if dataset is None:
            raise ValueError('Must provide a study name when running in single mode')
        if dataset == 'kddcup99':
            x_train, y_train, x_val, y_val, x_test, y_test = download_kddcup99_dataset()
        elif dataset == 'hepmass':
            x_train, y_train, x_val, y_val, x_test, y_test = download_hepmass_dataset()

        study_id = f"{'kddcup99'}:{study_tag}"

        study(ada_boost_decision_tree_study, f'Ada Boost Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(artificial_neural_networks_study, f'Artificial Neural Networks {study_id}', n_trials=n_trials,
              storage=storage)
        study(ada_boost_decision_tree_study, f'Ada Boost Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(decision_tree_study, f'Decision Tree {study_id}', n_trials=n_trials, storage=storage)
        study(k_nearest_neighbors_study, f'K Nearest Neighbors {study_id}', n_trials=n_trials, storage=storage)
        study(support_vector_machine_study, f'Support Vector Machine {study_id}', n_trials=n_trials, storage=storage)


if __name__ == '__main__':
    main()
