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
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output = 'pandas')
pd.options.mode.chained_assignment = None
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

desc = pd.DataFrame(index = list(train))
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['type'] = train.dtypes
desc = pd.concat([desc, train.describe().T], axis = 1)
desc
#%%

desc = pd.DataFrame(index = list(test))
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['type'] = test.dtypes
desc = pd.concat([desc, test.describe().T], axis = 1)
desc
# %%

# grouping the categorical and numerical features.
temporal_features = ['date_id', 'seconds_in_bucket']
categorical_features = ['imbalance_buy_sell_flag', 'stock_id']
numerical_features = train.drop(temporal_features + categorical_features + ['target'], axis = 1).columns
# %%
# Distribution of Numerical Features
fig, ax = plt.subplots(5, 2, figsize = (15, 20), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(numerical_features):
    
    sns.kdeplot(train[column], ax=ax[i], color=pal[0], fill = True)
    
    ax[i].set_title(f'{column} Distribution', size = 14)
    ax[i].set_xlabel(None)
    
fig.suptitle('Distribution of Numerical Features\nper Dataset\n', fontsize = 24, fontweight = 'bold')
plt.tight_layout()
# %%

# Distribution of Categorical Features
fig, ax = plt.subplots(1, 2, figsize = (20, 10), dpi = 300)
ax = ax.flatten()

ax[0].pie(
    train['imbalance_buy_sell_flag'].value_counts(), 
    shadow = True, 
    explode = [.1 for i in range(train['imbalance_buy_sell_flag'].nunique())], 
    autopct = '%1.f%%',
    textprops = {'size' : 14, 'color' : 'white'}
)

sns.countplot(data = train, y = 'imbalance_buy_sell_flag', ax = ax[1], palette = 'viridis', order = train['imbalance_buy_sell_flag'].value_counts().index)
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count in Train', fontsize = 15)
ax[1].set_ylabel('Imbalance Flag', fontsize = 15)
plt.xticks(fontsize = 12)

fig.suptitle('Distribution of Imbalance Flag\nin Train Dataset\n\n\n\n', fontsize = 30, fontweight = 'bold')
plt.tight_layout()
# %%
#Target Distribution

plt.figure(figsize = (20, 10), dpi = 300)

sns.kdeplot(train.target, fill = True)

plt.title('Target Distribution', weight = 'bold', fontsize = 30)
plt.show()
# %%
# Target Over Time

plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(data = train, x = 'date_id', y = 'target', hue = 'imbalance_buy_sell_flag', errorbar = None, palette = 'viridis')

plt.title('Average Target Over Days', weight = 'bold', fontsize = 30)
plt.show()
#%%

plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(data = train, x = 'seconds_in_bucket', y = 'target', hue = 'imbalance_buy_sell_flag', errorbar = None, palette = 'viridis')

plt.title('Average Target Over Seconds in Buckets', weight = 'bold', fontsize = 30)
plt.show()

#%%
###############################################################################################################################################################
#Preparation
# This is where we start preparing everything if we want to start building machine learning models. 
# we will use Time Series Split for our cross validation process. 
# We will also drop any rows where the missing target values are located.

X = train[~train.target.isna()]
X_test = test[~train.target.isna()]

X.info()
print(X.head())

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical = ['imbalance_size', 'reference_price', 'matched_size',
            'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size']

X_scaled = X.copy(deep=True)
X_scaled.loc[:, numerical] =  scaler.fit_transform(X.loc[:, numerical])

X_test_scaled = X_test.copy(deep=True)
X_test_scaled.loc[:, numerical] =  scaler.fit_transform(X_test.loc[:, numerical])
X_test_scaled.head(10)

#%%

#%%
kf = KFold(n_splits=10)
seed = 42
tss = TimeSeriesSplit(10)

os.environ['PYTHONHASHSEED'] = '42'
tf.keras.utils.set_random_seed(seed)

# %%
# Cross-Validation

def cross_val_score(estimator, save_path, cv = kf, label = ''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    #X = train[~train.target.isna()]
    #y = X.pop('target')
    
    X = X_scaled
    y = X.pop('target')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    #train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    
    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0
    
    #training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
        model = clone(estimator)
        
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        
        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        #train model
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, f'./{save_path}/{label}_{fold}.model')
        
        #make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
                  
        val_predictions[val_idx] += val_preds
        
        #evaluate model for a fold
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)
        
        # Update best model
        if best_val_score == 0 or val_score < best_val_score:
            best_val_score = val_score
            best_model_train_score = train_score
            best_model = model
            best_fold = fold
            
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')
    # Print best model score
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}')
    joblib.dump(best_model, f'./{save_path}/best_model.model')
    
    return val_scores, val_predictions
#%%


#%%
import joblib

model_save_path = "initial_run_kf_scaler"
models = [
   ('XGBoost', XGBRegressor(random_state = seed, objective = 'reg:absoluteerror', tree_method = 'hist', device = 'gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
           # ImbalanceCalculator,
            model
        ),
        save_path=model_save_path,
        label = label
    )
# %%
# Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`

# After testing, this error is cause by imbalance future features from ImbalanceCalculator. no inf or large value in train dataset.



# Check for infinite values
inf_in_train = train.isin([np.inf, -np.inf]).any()
inf_in_test = test.isin([np.inf, -np.inf]).any()

print(inf_in_train)
print(inf_in_test)

#%%

# Check for missing values in the train dataset
missing_in_train = train.isnull().any()

# Check for missing values in the test dataset
missing_in_test = test.isnull().any()

# Print the results
print("Missing values in train dataset:\n", missing_in_train)
print("\nMissing values in test dataset:\n", missing_in_test)

# %%
train.max()
#why such large values might occur. 
# imbalance_size             2.982028e+09
#  matched_size               7.713682e+09

# Potential Solution :
# 1. logarithmic transformation for these large monetary values. However, may negatively impact the interpretability of your model's results.

# 2. normalization techniques like Min-Max Scaling or Standard Scaling (Z-score normalization)

# 3. Adjusting XGBoost parameters such as max_depth, lambda (L2 regularization), alpha (L1 regularization), or max_delta_step 
# can help the model better handle the range of values in your data.

# 4. creating additional features that might capture the information in these columns more effectively.  
# categorizing these monetary values into different ranges or bands could be usefu


# %%

# %%

# logarithmic transformation to the columns with large values:

#train['matched_size'] = np.log1p(train['matched_size'])
#train['imbalance_size'] = np.log1p(train['imbalance_size'])

#train.max()

#%%

#train['ask_size'] = np.log1p(train['ask_size'])
#train['bid_size'] = np.log1p(train['bid_size'])


#train.max()
#%%



#%%

# Importance XGB
import xgboost as xgb
import joblib

X_train = train[~train.target.isna()]
y_train = X.pop('target')

model_xgb = XGBRegressor(random_state = seed, objective = 'reg:absoluteerror', tree_method = 'auto')
model_xgb.fit(X_train, y_train) 

# Save the trained xgboost model to a file
joblib.dump(model_xgb, 'xgboost_model.pkl')

#%%
loaded_model_xgb = joblib.load('xgboost_model.pkl')

xgb.plot_importance(loaded_model_xgb, importance_type='weight')
plt.show()