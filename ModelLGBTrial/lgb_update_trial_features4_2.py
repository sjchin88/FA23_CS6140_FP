# %%[markdown]
# All required import statement
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history

sns.set_theme(style='white', palette='viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None
seed = 42
tss = TimeSeriesSplit(10)
kf = KFold(n_splits=10)
os.environ['PYTHONHASHSEED'] = '42'

# %%[markdown]
# Prepare the function to read the data
dtypes = {
    'stock_id': np.uint8,
    'date_id': np.uint16,
    'seconds_in_bucket': np.uint16,
    'imbalance_buy_sell_flag': np.int8,
    'time_id': np.uint16,
}


def process_target(data):
    data.sort_values(by=['stock_id', 'time_id'], inplace=True)
    data['target_wap'] = data['wap'].shift(6) / data['wap'] * 10000
    data['target_wap'] = data.apply(lambda x: x['target_wap'] if
                                    x['seconds_in_bucket'] <= 480 else None,  axis=1)
    data['target_index'] = data.apply(lambda x: x['target_wap'] - x['target'] if
                                      x['seconds_in_bucket'] <= 480 else None, axis=1)
    data.info()
    data.head(10)
    data.tail(10)
    data.dropna(subset=['target_wap', 'target_index'], inplace=True)
    data.sort_values(by=['time_id', 'stock_id'], inplace=True)
    return data

# This will prepare the imbalance feature


def imbalance_calculator(x):

    x_copy = x.copy()

    x_copy['imb_s1'] = x.eval('(bid_size - ask_size) / (bid_size + ask_size)')
    x_copy['imb_s2'] = x.eval(
        '(imbalance_size - matched_size) / (matched_size + imbalance_size)')

    prices = ['reference_price', 'far_price',
              'near_price', 'ask_price', 'bid_price', 'wap']

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            if i > j:
                x_copy[f'{a}_{b}_imb'] = x.eval(f'({a} - {b}) / ({a} + {b})')

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            for k, c in enumerate(prices):
                if i > j and j > k:
                    max_ = x[[a, b, c]].max(axis=1)
                    min_ = x[[a, b, c]].min(axis=1)
                    mid_ = x[[a, b, c]].sum(axis=1)-min_-max_
                    x_copy[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)

    return x_copy


def read_data(data_path: str):
    """Read the data from the train and test csv files, split them into the x (features) and y(target)

    Args:
        data_path (str): absolute save path for the train and test data set 

    Returns:
        X (dataframe): Independent features for training
        y (dataframe): dependent features for training  
        X_test (dataframe): Independent features for testing
    """
    # Load data from the save path
    train = pd.read_csv(f'{data_path}/train.csv',
                        dtype=dtypes).drop(['row_id'], axis=1)

    # Check the data set
    print("preview data set")
    train.info()
    print(train.head())
    print(train.tail())
    gc.collect()

    # preprocess data for two additional targets and split data into X and y
    X = train[~train.target.isna()]
    X = imbalance_calculator(X)
    X = process_target(X)
    X.drop('time_id', axis=1, inplace=True)
    print("data after processing")
    X.info()
    print(X.head())
    print(X.tail())
    y = X.pop('target')
    y_wap = X.pop('target_wap')
    y_index = X.pop('target_index')

    return X, y, y_wap, y_index


# %%[markdown]
# This section will prepare the cross_validation function
# cv = time series split of 10

def cross_validation(estimator_wap, estimator_idx, save_path, X, y, y_wap, y_index, cv=kf, label=''):
    """cross validation function

    Args:
        estimator (model): chosen model 
        save_path (str) : target directory to save the model
        X (dataframe): Independent features for training
        y_wap (dataframe): dependent features for training  
        y_index (dataframe): Independent features for testing
        cv (split, optional): split for the cross validation. Defaults to tss.
        label (str, optional): special label. Defaults to ''.

    Returns:
        _type_: _description_
    """
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

    # training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_wap)):

        model_wap = clone(estimator_wap)
        model_idx = clone(estimator_idx)

        # define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        y_wap_train = y_wap.iloc[train_idx]
        y_index_train = y_index.iloc[train_idx]

        # define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        y_wap_val = y_wap.iloc[val_idx]
        y_index_val = y_index.iloc[val_idx]

        # train model
        model_wap.fit(X_train, y_wap_train)
        model_idx.fit(X_train, y_index_train)

        # Save the model
        joblib.dump(model_wap, f'./{save_path}/wap_{fold}.model')
        joblib.dump(model_wap, f'./{save_path}/idx_{fold}.model')

        # make predictions
        train_wap_preds = model_wap.predict(X_train)
        val_wap_preds = model_wap.predict(X_val)
        train_idx_preds = model_idx.predict(X_train)
        val_idx_preds = model_idx.predict(X_val)

        train_preds = (train_wap_preds - train_idx_preds)
        val_preds = (val_wap_preds - val_idx_preds)

        # evaluate model for a fold
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)

        # Update best model
        if best_val_score == 0 or val_score < best_val_score:
            best_val_score = val_score
            best_model_train_score = train_score
            best_model_wap = model_wap
            best_model_idx = model_idx
            best_fold = fold

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    # This line print the average
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    # Print best model score
    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}, fold:{best_fold}')
    joblib.dump(best_model_wap, f'./{save_path}/best_model_wap.model')
    joblib.dump(best_model_idx, f'./{save_path}/best_model_idx.model')
    return None


# %%[markdown]
# Call the required function and run the model
model_wap = LGBMRegressor(
    random_state=seed, objective='mae', device_type='gpu')
model_idx = LGBMRegressor(
    random_state=seed, objective='mae', device_type='gpu')
X, y, y_wap, y_index = read_data(
    "D:/OneDrive/NEU/CS6140/optiver-trading-at-the-close")
model_save_path = "initial_run_featuresComboImb"
# %%
cross_validation(
    model_wap,
    model_idx,
    save_path=model_save_path,
    X=X,
    y=y,
    y_wap=y_wap,
    y_index=y_index,
)


# %%[markdown]
# Explore the model trained
# Job saved is a pipeline, the model is in second step
best_wap = joblib.load(f"./{model_save_path}/best_model_wap.model")
trained_model = best_wap
print(trained_model)

lgb.plot_importance(trained_model, importance_type="gain", figsize=(
    7, 8), precision=0, title="LightGBM model_wap Feature Importance (Gain)")
plt.show()
lgb.plot_importance(trained_model, importance_type="split", figsize=(
    7, 8), precision=0, title="LightGBM model_wap Feature Importance (Split)")
plt.show()

best_idx = joblib.load(f"./{model_save_path}/best_model_idx.model")
trained_model = best_idx
print(trained_model)

lgb.plot_importance(trained_model, importance_type="gain", figsize=(
    7, 8), precision=0, title="LightGBM model_idx Feature Importance (Gain)")
plt.show()
lgb.plot_importance(trained_model, importance_type="split", figsize=(
    7, 8), precision=0, title="LightGBM model_idx Feature Importance (Split)")
plt.show()

# %%
