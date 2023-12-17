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
#Preparation
# This is where we start preparing everything if we want to start building machine learning models. 
# we will use Time Series Split for our cross validation process. 
# We will also drop any rows where the missing target values are located.
model_save_path = "initial_run_kf_hp3"

X = train[~train.target.isna()]
y = X.pop('target')
X_test = test[~train.target.isna()]

X.info()
print(X.head())

#%%
kf = KFold(n_splits=10)
seed = 42
tss = TimeSeriesSplit(10)

os.environ['PYTHONHASHSEED'] = '42'
tf.keras.utils.set_random_seed(seed)

#%%
#Warning about early_stopping_rounds: The warning regarding early_stopping_rounds is due to a change in how XGBoost handles this parameter. 
# In recent versions of XGBoost, it's recommended to specify early_stopping_rounds when initializing the model (in the constructor) rather than in the fit method. 
# This change was made for better compatibility with scikit-learn


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

#%%
import joblib
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
        
joblib.dump(model, f'{model_save_path}/best_model.model')
#%%
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
plot_optimization_history(study)
plot_param_importances(study)


# %%
# Cross-Validation

def cross_val_score(estimator, save_path, cv = kf, label = ''):
    # Build the save path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    X = train[~train.target.isna()]
    y = X.pop('target')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    #train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    
    best_model = None
    best_model_train_score = 0
    best_val_score = 0
    
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

        # Predict and evaluate
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)


         # Save fold model
        joblib.dump(model, f'{save_path}/{label}_fold_{fold}.model')

        # Update best model
        if best_val_score == 0 or val_score < best_val_score:
            best_val_score = val_score
            best_model_train_score = train_score
            best_model = model
            
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'Average Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | AverageTrain Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
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

models = [
   ('XGBoost', XGBRegressor(random_state = seed, objective = 'reg:absoluteerror', tree_method = 'hist', device = 'gpu', missing=np.nan))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            model
        ),
        save_path=model_save_path,
        param_grid= param_grid,
        label = label
    )
# %%

import joblib
from xgboost import plot_importance
import matplotlib.pyplot as plt

best_model_filename = f'./{model_save_path}/best_model.model'

# Load the pipeline object from the file
pipeline = joblib.load(best_model_filename)

# Assuming that XGBRegressor is the final step in the pipeline, access it
# If there are multiple steps, replace 'xgbregressor' with the correct step name
best_model = pipeline.named_steps['xgbregressor']

# Now you can access the booster directly
feature_importance = best_model.get_booster().get_score(importance_type='weight')

# Plot feature importance
plot_importance(best_model, importance_type='weight')
plt.show()

# Print feature importances
for feature, importance in feature_importance.items():
    print(f"Feature: {feature}, Importance: {importance}")
    
    
    
#%%

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