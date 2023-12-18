"""
Class Name    : CS6140 Machine Learning 
Session       : Fall 2023 (Seattle)
Author        : Team B - Shiang Jin Chin
Last Update   : 12/17/2023
Description   : Contains all required code to run lightGBM 
"""
# %% [markdown]
# Instruction for use: Download the train.csv file from https://www.kaggle.com/competitions/optiver-trading-at-the-close/data.
# Place it in a directory, pass the directory path to the read_data function
#
# For lightGBM, option of pip install
# https://www.geeksforgeeks.org/how-to-install-lightgbm-on-windows/
#
# Or build the binaries for GPU, I used following guide from window
# https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#id18
# If you use cpu, then the device type should be set to cpu.
#
# All required import statement
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time as time
import os
import gc

from sklearn import set_config
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history
# Common setting used
pd.set_option('display.max_rows', 100)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None
seed = 42
tss = TimeSeriesSplit(10)
kf = KFold(n_splits=10)
data_path = "D:/OneDrive/NEU/CS6140/optiver-trading-at-the-close"

# %% [markdown]
# Part 1 - Preprocessing
#
# Prepare the functions required
dtypes = {
    'stock_id': np.uint8,
    'date_id': np.uint16,
    'seconds_in_bucket': np.uint16,
    'imbalance_buy_sell_flag': np.int8,
    'time_id': np.uint16,
}


def scale_data(X):
    """scale the data using standard scaler

    Args:
        X (dataframe): raw data to be scaled

    Returns:
        scaled_X (dataframe): data with selected columns scaled
    """
    scaler = StandardScaler()
    # Select columns to scale
    numerical = ['imbalance_size', 'reference_price', 'matched_size',
                 'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
    # print(numerical)
    scaled_X = X.copy(deep=True)
    scaled_X.loc[:, numerical] = scaler.fit_transform(X.loc[:, numerical])
    return scaled_X


def read_data(data_path, drop_features=[], apply_scale=False, process_na=0):
    """Read the data from the train csv files, split them into the x (features) and y(target)
    default will drop row_id, time_id

    Args:
        data_path (str): absolute save path for the train and test data set 
        drop_features (list[str]) : list of additional features to drop
        apply_scale (bool, optional) : whether to apply standard scaler to the data, default is False
        process_na (int, optional) : option to deal with na, 0 = do nothing, 1 = fill 0 , 2 = drop

    Returns:
        X (dataframe): Independent features for training
        y (dataframe): dependent features for training  
    """
    # Load data from the save path, 'row_id' and 'time_id' has no used during initial run, these are dropped
    train = pd.read_csv(f'{data_path}/train.csv',
                        dtype=dtypes).drop(['row_id', 'time_id'], axis=1)
    train.drop(drop_features, axis=1, inplace=True)
    # Check the data set
    train.info()
    print(train.head())
    gc.collect()

    # split data into X and y
    X = train[~train.target.isna()]
    if process_na == 1:
        X.fillna(0, inplace=True)
    elif process_na == 2:
        X.dropna(inplace=True)
    if apply_scale:
        X = scale_data(X)

    y = X.pop('target')

    # Check the data set
    print("after split")
    X.info()
    print(X.head())
    return X, y


def cross_validation(model,  X, y, cv=kf, saving=False, save_path=None, apply_pca=False, label='LightGBM'):
    """cross validation function

    Args:
        model (model): customized lightGBM model
        X (dataframe): Independent features for training
        y (dataframe): dependent features for training  
        cv (split, optional): split for the cross validation. Defaults to kfold.
        saving (bool, optional): whether you want to save your model, default is False
        save_path (str, optional) : target directory to save the model, defauls is None
        apply_pca (bool, optional) : whether to apply PCA, default is False
        label (str, optional): special label. Defaults to LightGBM.

    Returns:
        model: best model trained
    """
    # Build the save path if not exist
    if saving and save_path != None and not os.path.exists(save_path):
        os.makedirs(save_path)

    # initiate score lists and variables to store best results
    train_scores, val_scores = [], []
    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

    # training model, for the cross validation split
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        model = clone(model)

        # define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Add PCA processing
        # I use all components, as main purpose is to aid in decision boundary
        if apply_pca:
            pca = PCA()
            pca.fit(X_train)
            pca.fit(X_val)

        # train model
        model.fit(X_train, y_train)

        # Save the model
        if saving:
            joblib.dump(model, f'./{save_path}/{label}_{fold}.model')

        # make predictions on training and validation set
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        # evaluate model for a fold using MAE
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)

        # Update best model
        if best_val_score == 0 or val_score < best_val_score:
            best_val_score = val_score
            best_model_train_score = train_score
            best_model = model
            best_fold = fold

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    # Wrap up after the end of cross validation loop
    # This line print the average
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    # Print best model score
    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}, fold:{best_fold}')
    # Save the best model and return
    if saving:
        joblib.dump(best_model, f'./{save_path}/best_model.model')
    return best_model


def plot_feature_importance(trained_model):
    """Small helper function to print the feature importance value

    Args:
        trained_model (model): trained LightGBM model
    """
    print(trained_model)
    lgb.plot_importance(trained_model, importance_type="gain", figsize=(
        7, 8), precision=0, title="LightGBM Feature Importance (Gain)")
    plt.show()
    lgb.plot_importance(trained_model, importance_type="split", figsize=(
        7, 8), precision=0, title="LightGBM Feature Importance (Split)")
    plt.show()


# %% [markdown]
# First read the data
X, y = read_data(data_path)
# %% [markdown]
# Part 1 - First run, you can switch between gpu (need lightGBM built for gpu) or cpu
# Note for cpu it took about 100s, for gpu it is about 94s
#
# You can also try different cross-validation method, tss (time series split) or
# kf (kfold) , here we use tss, later code we will use kfold
#
# If you want to save the model , look up at the cross_validation args to input
# the save path, and turn the saving option on
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

start = time.time()
best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=tss
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')

# Plot the feature importance for the best model
plot_feature_importance(best_model)

# %% [markdown]
# Part 1 - Second run, Now we want to try the effect of standardscaler().
# Read the data using scaled function, and run the model.
# Should take about 111 s with GPU

start = time.time()
X, y = read_data(data_path, apply_scale=True)
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    label='LightGBM_scaled'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')
# plot_feature_importance(best_model)

# %% [markdown]
# Part 1 - Third run, Now we want to try the effect of PCA().
# We need to modify the read data function to clean up the NaN value.
# And modify the cross_validation function to apply PCA() after the split
# First try with fillna with 0, It took about 150-200s
start = time.time()
# X, y = read_data(data_path, process_na=1)

# Different na processing, this time drop all na, should be faster
X, y = read_data(data_path, process_na=2)
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    apply_pca=True,
    label='LightGBM_pca'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')

# %% [markdown]
# Part 2 - Perform Experiments with Feature Engineering
#
# First define additional functions required


def add_derivatives(data_raw, derivative_targets):
    """add the first and second derivatives for target

    Args:
        data_raw (dataframe): raw training data
        derivative_targets (list[str]): list of derivate targets

    Returns:
        dataframe: processed raw data
    """
    # sort by stock_id so we can get next 10 seconds movement
    data_raw.sort_values(by=['stock_id', 'time_id'], inplace=True)
    for target in derivative_targets:
        if target not in data_raw.columns:
            continue
        data_raw[f'first_derivative_{target}'] = data_raw['reference_price'] - \
            data_raw['reference_price'].shift(1)
        data_raw[f'first_derivative_{target}'] = data_raw.apply(
            lambda x: 0 if x['seconds_in_bucket'] == 0 else x[f'first_derivative_{target}'], axis=1)
        data_raw[f'sec_derivative_{target}'] = data_raw[f'first_derivative_{target}'] - \
            data_raw[f'first_derivative_{target}'].shift(1)
        data_raw[f'sec_derivative_{target}'] = data_raw.apply(
            lambda x: 0 if x['seconds_in_bucket'] <= 10 else x[f'sec_derivative_{target}'], axis=1)

    # sort the data back
    data_raw.sort_values(by=['time_id', 'stock_id'], inplace=True)
    return data_raw


def imbalance_calculator(x):
    """calculate the imbalance features

    Args:
        x (dataframe): raw training data

    Returns:
        dataframe: processed raw data
    """
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


def read_data_feature_engineering(data_path, drop_features=[], process_na=0, apply_derivative=False, derivative_targets=[], apply_imb=False):
    """Read the data from the train csv files, split them into the x (features) and y(target)

    Args:
        data_path (str): save path of the data
        drop_features (list[str]) : list of additional features to drop
        process_na (int, optional) : option to deal with na, 0 = do nothing, 1 = fill 0 , 2 = drop
        apply_derivative (bool, optional): _description_. Defaults to False.
        derivative_targets (list, optional): _description_. Defaults to [].
        apply_imb (bool, optional): _description_. Defaults to False.

    Returns:
        dataframe: X (dependent variables)
        dataframe: y (target)
    """
    # Load data from the save path
    train = pd.read_csv(f'{data_path}/train.csv',
                        dtype=dtypes)
    train.drop(drop_features, axis=1, inplace=True)

    # derivative is na sensitive, we need to apply it before any operation on na
    if apply_derivative:
        train = add_derivatives(train, derivative_targets)
    train.drop(['row_id', 'time_id'], axis=1, inplace=True)

    # Check the data set
    train.info()
    print(train.head())
    print(train.tail())
    gc.collect()

    # split data into X and y
    X = train[~train.target.isna()]
    if process_na == 1:
        X.fillna(0, inplace=True)
    elif process_na == 2:
        X.dropna(inplace=True)
    if apply_imb:
        X = imbalance_calculator(X)
    y = X.pop('target')

    X.info()
    print(X.head())
    return X, y


# %% [markdown]
# Part2 - First run, Start with dropping four least important feature
# Should take about 140s
start = time.time()
X, y = read_data(data_path, drop_features=['near_price', 'stock_id', 'far_price',
                                           'imbalance_buy_sell_flag'])
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    label='LightGBM_feature_1'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')
# %% [markdown]
# Part2 - Second run, perform derivatives

start = time.time()

# This will take longer, about 250s
X, y = read_data_feature_engineering(
    data_path, apply_derivative=True, derivative_targets=['reference_price'])

# More derivative targets, warning, this could take about 15 minutes
# X, y = read_data_feature_engineering(data_path, apply_derivative=True,
#                                     derivative_targets=['ask_size', 'bid_size', 'matched_size',
#                                                         'bid_price', 'ask_price', 'wap', 'reference_price', 'imbalance_size'])
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    label='LightGBM_feature_2'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')

# %% [markdown]
# Part2 - Third run, add imbalance features

start = time.time()

# This will take longer, about 325s
X, y = read_data_feature_engineering(data_path, apply_derivative=False, derivative_targets=[
                                     'reference_price'], apply_imb=True)

# with derivative targets, warning, this could take about 10-15 minutes
# X, y = read_data_feature_engineering(data_path, apply_derivative=True, derivative_targets=[
#                                     'reference_price'], apply_imb=True)
lightGBM_model = LGBMRegressor(
    random_state=seed,
    # mse was used as it was found to perform better compared to objective function of 'mae
    objective='mse',
    device_type='gpu',  # You can switch to cpu
    # <0 to show only Fatal msg, 0 for Error(Warning), 1 for info, > 1 for Debug
    verbosity=0
)

best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    label='LightGBM_feature_2'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')
# Dont forget to take a look at the feature importance
plot_feature_importance(best_model)

# %% [markdown]
# Part2 - Fourth run, try two model. Note this function is slightly different compared to the results
#
# in the report, for the report, model_wap and model_idx are trained together with same training data,
# and then used to predict the outcome of validation set. For submission, the whole training set is used to train
# both model
#
# In this approach, the original data is first split into training & testing set.
# The training set is further split into kfold (10), and used to train and find the best
# model_wap and model_idx separately.
# Then the best model_wap and model_idx are used to predict the target in the testing set.
#
# First define the functions required


def process_target(data):
    """process the data to compute target_wap and target_index

    Args:
        data (dataframe): raw data to be processed

    Returns:
        dataframe: processed data
    """

    data.sort_values(by=['stock_id', 'time_id'], inplace=True)
    data['target_wap'] = data['wap'].shift(6) / data['wap'] * 10000
    data['target_wap'] = data.apply(lambda x: x['target_wap'] if
                                    x['seconds_in_bucket'] <= 480 else None,  axis=1)
    data['target_index'] = data.apply(lambda x: x['target_wap'] - x['target'] if
                                      x['seconds_in_bucket'] <= 480 else None, axis=1)
    # Check if data processed correctly
    data.info()
    data.head(10)
    data.tail(10)
    data.dropna(subset=['target_wap', 'target_index'], inplace=True)
    data.sort_values(by=['time_id', 'stock_id'], inplace=True)
    return data


def read_data_2model(data_path: str, apply_imb=False):
    """Read the data from the train csv files, split them into the training and testing set
    # For training, computed the target_wap and target_idx

    Args:
        data_path (str): absolute save path for the train and test data set 
        apply_imb (bool, optional) : whether to apply the imbalance factor, default is false

    Returns:
        X (dataframe): Independent features for training
        y (dataframe): dependent features for training  
        y_wap (dataframe): dependent features for training  
        y_index (dataframe): dependent features for training  
        X_test (dataframe): independent features for testing 
        y_test (dataframe): dependentt features for testing 
    """
    # Load data from the save path
    data = pd.read_csv(f'{data_path}/train.csv',
                       dtype=dtypes).drop(['row_id'], axis=1)
    data.dropna(subset=['target'], inplace=True)
    if apply_imb:
        data = imbalance_calculator(data)

    # Split into train and test set
    from sklearn.model_selection import train_test_split
    train, X_test = train_test_split(
        data, test_size=0.1, train_size=0.9, shuffle=False)

    # Process the test set
    X_test.drop('time_id', axis=1, inplace=True)
    print("preview test set")
    X_test.info()
    print(X_test.head())
    print(X_test.isna().sum())
    y_test = X_test.pop('target')

    # Check the training data set
    print("\n\npreview data set")
    train.info()
    print(train.head())
    gc.collect()

    # preprocess data for two additional targets and split data into X and y
    X = process_target(train)
    X.drop('time_id', axis=1, inplace=True)

    # Check if data sorted correctly back
    print("data after processing")
    X.info()
    print(X.head())
    print(X.tail())
    y = X.pop('target')
    y_wap = X.pop('target_wap')
    y_index = X.pop('target_index')

    return X, y, y_wap, y_index, X_test, y_test


# %% [markdown]
# Now running the process. First get the data, should take about 60-70s
start = time.time()
X, y, y_wap, y_index, X_test, y_test = read_data_2model(data_path)

# Can also switch to apply imb features, training took longer
# X, y, y_wap, y_index, X_test, y_test = read_data_2model(
#    data_path, apply_imb=True)

end = time.time()
print(f'Data processing time took = {(end - start):.0f} seconds')
start = end

# Train and get the best model_wap overall will took about 220 s
print('\ntraining model_wap')
model_wap = LGBMRegressor(
    random_state=seed, objective='mse', device_type='gpu',  verbosity=0)
best_model_wap = cross_validation(model_wap, X, y_wap, label='modelwap')

# Train and get the best model_idx
print('\ntraining model_idx')
model_idx = LGBMRegressor(
    random_state=seed, objective='mse', device_type='gpu', verbosity=0)
best_model_idx = cross_validation(model_idx, X, y_index, label='modelidx')

test_wap_preds = best_model_wap.predict(X_test)
test_idx_preds = best_model_idx.predict(X_test)
test_target_preds = test_wap_preds - test_idx_preds

test_score = mean_absolute_error(y_test, test_target_preds)

end = time.time()
print(f'Training and testing time took = {(end - start):.0f} seconds')
print(f'MAE for two model approach is {test_score}')

# Dont forget to check the feature importance. You can also see the correlation in the EDA
print("\nFeature importance for model wap")
plot_feature_importance(best_model_wap)
print("\nFeature importance for model idx")
plot_feature_importance(best_model_idx)

# %% [markdown]
# Part3 - hyperparameter tuning
#
# This will set the Optuna objective function
# Set the objective function for Optuna study and running trials


def objective(trial, X, y):
    """objective function for optina

    Args:
        trial (func): represent a trial
        X (dataframe): independent variables
        y (dataframe): dependent variable

    Returns:
        float: objective score
    """
    max_depth = trial.suggest_int("max_depth", 4, 12)
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 2**(max_depth-1), 2**(max_depth), step=2**(max_depth-1)//8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 32, 256),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1)
    }
    cv = KFold(n_splits=5
               )
    cv_scores = np.empty(10)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = LGBMRegressor(random_state=seed, objective='mse',
                              device_type='gpu', verbosity=-1, early_stopping_rounds=50, **param_grid)
        # train model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  callbacks=[
                  LightGBMPruningCallback(trial, 'l2')])
        val_preds = model.predict(X_val)

        # evaluate model for a fold
        val_score = mean_absolute_error(y_val, val_preds)
        cv_scores[fold] = val_score
    return np.mean(cv_scores)


# %% [markdown]
# Read the data and Run the optuna function. warning, this could take very long
# To show the function is working, I set the n_trials value to a low value
# early_stopping_rounds to 50, cross validation fold for objective function to 5 instead 10.
# In actual i use a larger value for better result. This should take about 20 minutes
start = time.time()
X, y = read_data_feature_engineering(data_path, apply_derivative=False, derivative_targets=[
                                     'reference_price'], apply_imb=True)
study = optuna.create_study(direction='minimize', study_name='LGBM Regressor')
def func(trial): return objective(trial, X, y)


study.optimize(func, n_trials=5)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')

# Check the optimization history and param_importances
print(f"\tBest value (mse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
plot_optimization_history(study)
plot_param_importances(study)
# %% [markdown]
# Rerun the training using best hyperparameter obtained, could take very long (like 30 minutes as well)
lightGBM_model = LGBMRegressor(random_state=seed, objective='mse',
                               device_type='gpu', verbosity=-1, **study.best_params)
start = time.time()
best_model = cross_validation(
    model=lightGBM_model,
    X=X,
    y=y,
    cv=kf,
    label='LightGBM_feature_2'
)
end = time.time()
print(f'training time took = {(end - start):.0f} seconds')

# %% [markdown]
# End of this file
