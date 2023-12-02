# %%[markdown]
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
import lightgbm as lgb

# Make directory for save model path

os.system('mkdir models')
sns.set_theme(style='white', palette='viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None
# %%[markdown]
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

gc.collect()

# %%[markdown]
X = train[~train.target.isna()]
y = X.pop('target')

seed = 42
tss = TimeSeriesSplit(10)

os.environ['PYTHONHASHSEED'] = '42'

# %%[markdown]


def imbalance_calculator(x):

    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
                ]

    x_copy = x.copy()

    # imb_s1 and imb_s2 evaluate the imbalance_size
    x_copy['imb_s1'] = x.eval('(bid_size - ask_size) / (bid_size + ask_size)')
    x_copy['imb_s2'] = x.eval(
        '(imbalance_size - matched_size) / (matched_size + imbalance_size)')

    prices = ['reference_price', 'far_price',
              'near_price', 'ask_price', 'bid_price', 'wap']

    # imbalance based on a pair of prices
    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            if i > j:
                x_copy[f'{a}_{b}_imb'] = x.eval(f'({a} - {b}) / ({a} + {b})')
                features.append(f'{a}_{b}_imb')

    # imbalance based on a triplet of prices
    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            for k, c in enumerate(prices):
                if i > j and j > k:
                    max_ = x[[a, b, c]].max(axis=1)
                    min_ = x[[a, b, c]].min(axis=1)
                    mid_ = x[[a, b, c]].sum(axis=1)-min_-max_

                    x_copy[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')

    return x_copy[features]


ImbalanceCalculator = FunctionTransformer(imbalance_calculator)
# %%[markdown]

# cv = time series split of 10
# What does time series split do?


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
        joblib.dump(model, f'./models/{label}_{fold}.model')

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
    ('LightGBM', LGBMRegressor(random_state=seed, objective='mae', device_type='gpu'))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            ImbalanceCalculator,
            model
        ),
        label=label
    )


# %%[markdown]
# Explore the model trained
# Job saved is a pipeline, the model is in second step
pipeline = joblib.load("./models/best_model.model")
trained_model = pipeline[1]
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
