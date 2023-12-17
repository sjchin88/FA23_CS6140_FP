#%%[markdown]

import tensorflow as tf
print(tf.__version__)
#%%
# Loading Libraries and Datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import gc

from sklearn import set_config
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import optuna

# %%
dtypes = {
    'stock_id' : np.uint8,
    'date_id' : np.uint16,
    'seconds_in_bucket' : np.uint16,
    'imbalance_buy_sell_flag' : np.int8,
    'time_id' : np.uint16,
}

train = pd.read_csv(r'optiver-trading-at-the-close/train.csv', dtype = dtypes).drop(['row_id', 'time_id'], axis = 1)
test = pd.read_csv(r'optiver-trading-at-the-close/example_test_files/test.csv', dtype = dtypes).drop(['row_id', 'time_id'], axis = 1)

gc.collect()
#%%

train.head(10)

#%%

#%%
###############################################################################################################################################################
X = train[~train.target.isna()]
y = X.pop('target')
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

from xgboost import XGBRegressor

# Initialize the XGBRegressor
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f'Mean Absolute Error for XGBoost: {xgb_mae}')



#%%
import lightgbm as lgb

# Initialize the LGBMRegressor
lgb_model = lgb.LGBMRegressor(random_state=42)

# Fit the model
lgb_model.fit(X_train, y_train)

# Predictions
lgb_predictions = lgb_model.predict(X_test)

# Evaluate the model
lgb_mae = mean_absolute_error(y_test, lgb_predictions)
print(f'Mean Absolute Error for LightGBM: {lgb_mae}')

# %%
