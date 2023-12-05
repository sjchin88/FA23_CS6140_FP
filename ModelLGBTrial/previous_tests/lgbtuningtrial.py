# %%[markdown]
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow as tf
import os
import gc

from sklearn import set_config
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb

# Make directory for save model path


sns.set_theme(style='white', palette='viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None

# %%[markdown]
# Read the data 

dtypes = {
    'stock_id': np.uint8,
    'date_id': np.uint16,
    'seconds_in_bucket': np.uint16,
    'imbalance_buy_sell_flag': np.int8,
    'time_id': np.uint16,
}

train = pd.read_csv('D:/OneDrive/NEU/CS6140/optiver-trading-at-the-close/train.csv',
                    dtype=dtypes).drop(['row_id', 'time_id'], axis=1)
test = pd.read_csv('D:/OneDrive/NEU/CS6140/optiver-trading-at-the-close/test.csv',
                   dtype=dtypes).drop(['row_id', 'time_id'], axis=1)
train.info()
print(train.head())
gc.collect()

X = train[~train.target.isna()]
y = X.pop('target')

X.info()
print(X.head())
seed = 42
tss = TimeSeriesSplit(10)

os.environ['PYTHONHASHSEED'] = '42'

# %%[markdown]
# Set the objective function for Optuna study and running trials
def objective(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3)
    }
    cv = tss
    cv_scores = np.empty(10)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = LGBMRegressor(random_state=seed, objective='mse',
                              device_type='gpu', early_stopping_rounds=100, **param_grid)
        # train model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  callbacks=[
                  LightGBMPruningCallback(trial, 'l2')])
        val_preds = model.predict(X_val)

        # evaluate model for a fold
        val_score = mean_absolute_error(y_val, val_preds)
        cv_scores[fold] = val_score
    return np.mean(cv_scores)


study = optuna.create_study(direction='minimize', study_name='LGBM Regressor')
def func(trial): return objective(trial, X, y)


study.optimize(func, n_trials=100)




# %%[markdown]
# Check the optimization history and param_importances
print(f"\tBest value (mse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
plot_optimization_history(study)
plot_param_importances(study)



# %%[markdown]
# cv = time series split of 10
# What does time series split do?
if not os.path.exists("models_tuned"):
    os.makedirs("models_tuned")


def cross_val_score(estimator, cv=tss, label=''):

    X = train[~train.target.isna()]
    y = X.pop('target')

    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    best_model = None
    best_model_train_score = 0
    best_val_score = 0

    # training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        model = clone(estimator)

        # define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # train model
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, f'./models_tuned/{label}_{fold}.model')

        # make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        val_predictions[val_idx] += val_preds

        # evaluate model for a fold
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)

        # Update best model
        if best_val_score == 0 or val_score < best_val_score:
            best_val_score = val_score
            best_model_train_score = train_score
            best_model = model

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    # This line print the average
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    # Print best model score
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}')
    joblib.dump(best_model, f'./models/best_model.model')

    return val_scores, val_predictions


# %%
models = [
    ('LightGBM', LGBMRegressor(random_state=seed, objective='mse', device_type='gpu', learning_rate=study.best_params['learning_rate'], n_estimators=study.best_params['n_estimators']))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            model
        ),
        label=label
    )


# %%[markdown]
# Explore the model trained
# Job saved is a pipeline, the model is in second step
pipeline = joblib.load("./models/best_model.model")
trained_model = pipeline[0]
print(trained_model)

lgb.plot_importance(trained_model, importance_type="gain", figsize=(
    7, 8), precision=0, title="LightGBM Feature Importance (Gain)")
plt.show()
# feature_importance = model.feature_importances_
# Get column name as list for feature
# feature_names = list(train.columns.values)
# gain_importance_df = pd.DataFrame({'Feature':feature_names, 'Gain': feature_importance})
# print(gain_importance_df.sort_values(by='Gain', ascending=False))
lgb.plot_importance(trained_model, importance_type="split", figsize=(
    7, 8), precision=0, title="LightGBM Feature Importance (Split)")
plt.show()

# %%
