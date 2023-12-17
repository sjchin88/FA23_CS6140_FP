"""
Class Name    : CS6140 Machine Learning 
Session       : Fall 2023 (Seattle)
Author        : Team B - Jason Gao
Last Update   : 12/17/2023
Description   : Contains all required code to run XGBoost 
"""

# %%[markdown]
# Instruction to run:
# Install missing package
# Download data from optiver and replace tha data path

from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
import optuna
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone
from sklearn import set_config
import gc
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
print(tf.__version__)
# %%
# Loading Libraries and Datasets


# %%

dtypes = {
    'stock_id': np.uint8,
    'date_id': np.uint16,
    'seconds_in_bucket': np.uint16,
    'imbalance_buy_sell_flag': np.int8,
    'time_id': np.uint16,
}

train = pd.read_csv(r'optiver-trading-at-the-close/train.csv',
                    dtype=dtypes).drop(['row_id', 'time_id'], axis=1)
test = pd.read_csv(r'optiver-trading-at-the-close/example_test_files/test.csv',
                   dtype=dtypes).drop(['row_id', 'time_id'], axis=1)

gc.collect()
# %%

train.head(10)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None
# %%
################################################################################################################################################################
# EDA
desc = pd.DataFrame(index=list(train))
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['type'] = train.dtypes
desc = pd.concat([desc, train.describe().T], axis=1)
desc
# %%

desc = pd.DataFrame(index=list(test))
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['type'] = test.dtypes
desc = pd.concat([desc, test.describe().T], axis=1)
desc
# %%

# grouping the categorical and numerical features.
temporal_features = ['date_id', 'seconds_in_bucket']
categorical_features = ['imbalance_buy_sell_flag', 'stock_id']
numerical_features = train.drop(
    temporal_features + categorical_features + ['target'], axis=1).columns
# %%
# Distribution of Numerical Features
sns.set_theme(style='white', palette='viridis')
pal = sns.color_palette('viridis')


fig, ax = plt.subplots(5, 2, figsize=(15, 20), dpi=300)
ax = ax.flatten()

for i, column in enumerate(numerical_features):

    sns.kdeplot(train[column], ax=ax[i], color=pal[0], fill=True)

    ax[i].set_title(f'{column} Distribution', size=14)
    ax[i].set_xlabel(None)

fig.suptitle('Distribution of Numerical Features\nper Dataset\n',
             fontsize=24, fontweight='bold')
plt.tight_layout()
# %%

# Distribution of Categorical Features
fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
ax = ax.flatten()

ax[0].pie(
    train['imbalance_buy_sell_flag'].value_counts(),
    shadow=True,
    explode=[.1 for i in range(train['imbalance_buy_sell_flag'].nunique())],
    autopct='%1.f%%',
    textprops={'size': 14, 'color': 'white'}
)

sns.countplot(data=train, y='imbalance_buy_sell_flag',
              ax=ax[1], palette='viridis', order=train['imbalance_buy_sell_flag'].value_counts().index)
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize=12)
ax[1].set_xlabel('Count in Train', fontsize=15)
ax[1].set_ylabel('Imbalance Flag', fontsize=15)
plt.xticks(fontsize=12)

fig.suptitle('Distribution of Imbalance Flag\nin Train Dataset\n\n\n\n',
             fontsize=30, fontweight='bold')
plt.tight_layout()
# %%
# Target Distribution

plt.figure(figsize=(20, 10), dpi=300)

sns.kdeplot(train.target, fill=True)

plt.title('Target Distribution', weight='bold', fontsize=30)
plt.show()
# %%
# Target Over Time

plt.figure(figsize=(20, 10), dpi=300)

sns.lineplot(data=train, x='date_id', y='target',
             hue='imbalance_buy_sell_flag', errorbar=None, palette='viridis')

plt.title('Average Target Over Days', weight='bold', fontsize=30)
plt.show()
# %%

plt.figure(figsize=(20, 10), dpi=300)

sns.lineplot(data=train, x='seconds_in_bucket', y='target',
             hue='imbalance_buy_sell_flag', errorbar=None, palette='viridis')

plt.title('Average Target Over Seconds in Buckets', weight='bold', fontsize=30)
plt.show()
# %%
###############################################################################################################################################################
# basline model
X = train[~train.target.isna()]
y = X.pop('target')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%

# Initialize the XGBRegressor
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f'Mean Absolute Error for XGBoost: {xgb_mae}')


xgb.plot_importance(xgb_model, importance_type='weight')
plt.show()

# %%

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

# Check for infinite values
inf_in_train = train.isin([np.inf, -np.inf]).any()
inf_in_test = test.isin([np.inf, -np.inf]).any()

print(inf_in_train)
print(inf_in_test)

# %%

# Check for missing values in the train dataset
missing_in_train = train.isnull().any()

# Check for missing values in the test dataset
missing_in_test = test.isnull().any()

# Print the results
print("Missing values in train dataset:\n", missing_in_train)
print("\nMissing values in test dataset:\n", missing_in_test)
# %%

seed = 42
tss = TimeSeriesSplit(10)
os.environ['PYTHONHASHSEED'] = '42'
tf.keras.utils.set_random_seed(seed)

# %%
# tss Cross-Validation


def cross_val_score(estimator, save_path, cv=tss, label=''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = train[~train.target.isna()]
    y = X.pop('target')

    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []

    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

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
        joblib.dump(model, f'./{save_path}/{label}_{fold}.model')

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
            best_fold = fold

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    print(f'Average Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Avergae Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')

    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')
    # Print best model score
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}')
    joblib.dump(best_model, f'./{save_path}/best_model.model')

    return val_scores, val_predictions


# %%
model_save_path = "initial_run_tss"
models = [
    ('XGBoost', XGBRegressor(random_state=seed, objective='reg:absoluteerror',
     tree_method='hist', device='gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            # ImbalanceCalculator,
            model
        ),
        save_path=model_save_path,
        label=label
    )

# %%
# %%
kf = KFold(n_splits=10)
# k-folder Cross-Validation


def cross_val_score(estimator, save_path, cv=kf, label=''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = train[~train.target.isna()]
    y = X.pop('target')

    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []

    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

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
        joblib.dump(model, f'./{save_path}/{label}_{fold}.model')

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
            best_fold = fold

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    print(f'Average Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Avergae Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')

    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')
    # Print best model score
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}')
    joblib.dump(best_model, f'./{save_path}/best_model.model')

    return val_scores, val_predictions


# %%
model_save_path = "initial_run_kf"
models = [
    ('XGBoost', XGBRegressor(random_state=seed, objective='reg:absoluteerror',
     tree_method='hist', device='gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            # ImbalanceCalculator,
            model
        ),
        save_path=model_save_path,
        label=label
    )
# %%

# StandardScaler

X = train[~train.target.isna()]
X_test = test[~train.target.isna()]

X.info()
print(X.head())
# %%

X.head(10)
# %%

scaler = StandardScaler()
numerical = ['imbalance_size', 'reference_price', 'matched_size',
             'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size']

X_scaled = X.copy(deep=True)
X_scaled.loc[:, numerical] = scaler.fit_transform(X.loc[:, numerical])
X_scaled.head(10)
# %%
# Cross-Validation
# change cv=tss to test tss


def cross_val_score(estimator, save_path, cv=kf, label=''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = X_scaled
    y = X_scaled.pop('target')
    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []

    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

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
        joblib.dump(model, f'./{save_path}/{label}_{fold}.model')

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
            best_fold = fold

        # append model score for a fold to list
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
# %%


# %%

model_save_path = "initial_run_kf_scaler"
models = [
    ('XGBoost', XGBRegressor(random_state=seed, objective='reg:absoluteerror',
     tree_method='hist', device='gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            # ImbalanceCalculator,
            model
        ),
        save_path=model_save_path,
        label=label
    )

# %%
# Check for NaN values in each column
nan_counts = X_scaled.isna().sum()

# Print the count of NaN values in each column
print("nan_counts :", nan_counts)
# %%


X = train[~train.target.isna()]

scaler = StandardScaler()
numerical = ['imbalance_size', 'reference_price', 'matched_size',
             'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size']

X_scaled = X.copy(deep=True)
X_scaled.loc[:, numerical] = scaler.fit_transform(X.loc[:, numerical])
X_scaled.head(10)
# Assuming X is your data
# Impute NaN values with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)
X_imputed.head(10)
# %%

# Apply PCA
pca = PCA()
pca.fit(X_imputed)
# cumulative explained variance ratio
cvr = np.cumsum(pca.explained_variance_ratio_)

print("cvr:", cvr)
# Choose the minimum number of principal components for 95% variance
num_components = np.argmax(cvr >= 0.95) + 1

print("Number of principal components for variance retention:", num_components)


# %%

pca = PCA(3)
X_transformed = pca.fit_transform(X_imputed)

# Check the number of features in the transformed dataset
num_features_transformed = X_transformed.shape[1]
print("the number of features in the transformed dataset: ",
      num_features_transformed)

X_transformed.head(10)

# %%
# Cross-Validation


def cross_val_score(estimator, save_path, cv=kf, label=''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # X = train[~train.target.isna()]
    # y = X.pop('target')

    X = X_transformed
    y = X_imputed.pop('target')

    # initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    # train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []

    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    best_fold = 0

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
        joblib.dump(model, f'./{save_path}/{label}_{fold}.model')

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
            best_fold = fold

        # append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)

    for fold in range(len(val_scores)):
        print(
            f'fold:{fold}, Val Score: {val_scores[fold]}, Train Score: {train_scores[fold]}')

    print(f'Average Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Average Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    # Print best model score
    print(
        f'Best validation score: {best_val_score}, associated train score: {best_model_train_score}')
    joblib.dump(best_model, f'./{save_path}/best_model.model')

    return val_scores, val_predictions
# %%


# %%

model_save_path = "initial_run_impute_kf_PCA3"
models = [
    ('XGBoost', XGBRegressor(random_state=seed, objective='reg:absoluteerror',
     tree_method='hist', device='gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            # ImbalanceCalculator,
            model
        ),
        save_path=model_save_path,
        label=label
    )
# %%
# %%
# Warning about early_stopping_rounds: The warning regarding early_stopping_rounds is due to a change in how XGBoost handles this parameter.
# In recent versions of XGBoost, it's recommended to specify early_stopping_rounds when initializing the model (in the constructor) rather than in the fit method.
# This change was made for better compatibility with scikit-learn
X = train[~train.target.isna()]
y = X.pop('target')

# Optuna Objective Function


def objective(trial):
    param_grid = {
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': trial.suggest_int("n_estimators", 50, 400),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'max_depth': trial.suggest_int("max_depth", 2, 9),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 9),
        'gamma': trial.suggest_float("gamma", 0, 0.5),
        'subsample': trial.suggest_float("subsample", 0.5, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        'early_stopping_rounds': 100,  # Add this line
    }
    model = XGBRegressor(random_state=seed, **param_grid)
    kf = KFold(n_splits=10)
    val_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds = model.predict(X_val)
        val_score = mean_absolute_error(y_val, val_preds)
        val_scores.append(val_score)

    return np.mean(val_scores)


# Run Optuna Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)


# Train the best model
best_params = study.best_params
best_params['tree_method'] = 'hist'
best_params['device'] = 'cuda'
model = XGBRegressor(random_state=seed, **best_params)
model.fit(X, y)

# %%
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

joblib.dump(model, f'{model_save_path}/best_model.model')
# %%

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
plot_optimization_history(study)
plot_param_importances(study)

# %%
# best tuning so far
'''
    n_estimators=343,
    learning_rate=0.027946309330522237,
    max_depth=9,
    min_child_weight=4,
    gamma=0.1871870798949814,
    subsample=0.9376967284854023,
    colsample_bytree=0.7570653610388687,
    reg_alpha=0.13271062560898594,
    reg_lambda=0.010619306237188686,
'''
