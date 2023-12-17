"""
Class Name    : CS6140 Machine Learning 
Session       : Fall 2023 (Seattle)
Author        : Team B - YiShuang Chen & Shiang Jin, Chin
Last Update   : 12/17/2023
Description   : Contains all required code to run EDA Analysis and Linear Regression
"""
# %% [markdown]
# Instruction for use: Download the train.csv file from https://www.kaggle.com/competitions/optiver-trading-at-the-close/data.
# Place it in a directory, pass the directory path to the read_csv function

###
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import statsmodels.api as sm
import time as time
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# This datapath is for us, change it to your place
data_path = "D:/OneDrive/NEU/CS6140/optiver-trading-at-the-close"
data = pd.read_csv(f'{data_path}/train.csv')
data = data.dropna()

# %% [markdown]
# This section plot some histogram
min_value = data['ask_size'].min()
max_value = data['ask_size'].max()


data['ask_size'].hist(range=(min_value, max_value))
plt.title('Histogram of ask_size')
plt.xlabel('ask_size')
plt.ylabel('Frequency')
plt.show()


data.boxplot(column=['ask_size'])
plt.title('Boxplot of ask_size')
plt.ylabel('ask_size')
plt.show()

variables = ['ask_size', 'matched_size', 'imbalance_size', 'far_price', 'near_price',
             'reference_price', 'bid_price', 'ask_price', 'wap', 'bid_size']


for var in variables:
    min_value = data[var].min()
    max_value = data[var].max()

    data[var].hist(range=(min_value, max_value))
    plt.title('Histogram of ' + var)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

    data.boxplot(column=[var])
    plt.title('Boxplot of ' + var)
    plt.ylabel(var)
    plt.show()

# %% [markdown]
# This section plot some violin plots


variables = ['ask_size', 'matched_size', 'imbalance_size', 'far_price', 'near_price',
             'reference_price', 'bid_price', 'ask_price', 'wap', 'bid_size']

for var in variables:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x=data[var])
    plt.title('Violin plot of ' + var)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# %% [markdown]
# This section plot explore the target, range and distribution


# Load the dataset
data = pd.read_csv(f'{data_path}/train.csv')
data.head()

# Assuming 'Close' is the target variable
target = data['target']

# Calculating the range of the target variable
range_min = target.min()
range_max = target.max()
target_range = range_max - range_min

print(
    f"Range of the target variable: {target_range} (Min: {range_min}, Max: {range_max})")

# Get the 0.5% quantile and 99.5% quantile, where 99% of data is enclosed
min_value = target.quantile(0.005)
max_value = target.quantile(0.995)

# 绘制直方图
target.hist(range=(min_value, max_value), bins=20)
plt.title('Histogram for 99% of target distribution')
plt.xlabel('target')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# This section look at the overall datasets,
# Get the number of rows and columns
rows, columns = data.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")


# Count unique stock_id
unique_stock_id_count = data['stock_id'].nunique()

# Count unique date_id
unique_date_id_count = data['date_id'].nunique()

# Count unique seconds_in_bucket
unique_seconds_in_bucket_count = data['seconds_in_bucket'].nunique()

print(f"Unique stock_id count: {unique_stock_id_count}")
print(f"Unique date_id count: {unique_date_id_count}")
print(f"Unique seconds_in_bucket count: {unique_seconds_in_bucket_count}")

print("after dropping all na value")
data = data.dropna()

rows, columns = data.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")


# Count unique stock_id
unique_stock_id_count = data['stock_id'].nunique()

# Count unique date_id
unique_date_id_count = data['date_id'].nunique()

# Count unique seconds_in_bucket
unique_seconds_in_bucket_count = data['seconds_in_bucket'].nunique()

print(f"Unique stock_id count: {unique_stock_id_count}")
print(f"Unique date_id count: {unique_date_id_count}")
print(f"Unique seconds_in_bucket count: {unique_seconds_in_bucket_count}")

# %% [markdown]
# This section perform pearsonr and spearmanr correlation analysis


df = data

float_columns = df.select_dtypes(include=['float']).columns


def pearson_correlation(df, target_column, float_columns):
    correlations = {}
    for column in float_columns:
        if column != target_column:
            corr, _ = pearsonr(df[column], df[target_column])
            correlations[column] = corr
    return correlations


def spearman_correlation(df, target_column, float_columns):
    correlations = {}
    for column in float_columns:
        if column != target_column:
            corr, _ = spearmanr(df[column], df[target_column])
            correlations[column] = corr
    return correlations


pearson_results = pearson_correlation(df, 'target', float_columns)
spearman_results = spearman_correlation(df, 'target', float_columns)

print("pearson_correlation:", pearson_results)
print("spearman_correlation:", spearman_results)

sorted_pearson = sorted(pearson_results.items(),
                        key=lambda x: x[1], reverse=True)
sorted_spearman = sorted(spearman_results.items(),
                         key=lambda x: x[1], reverse=True)

print("Sorted Pearson Correlation:", sorted_pearson)
print("Sorted Spearman Correlation:", sorted_spearman)


# %% [markdown]
# We will do some ols analysis

data = pd.read_csv(f'{data_path}/train.csv')
data.info()

start = time.time()
# time_id & row_id has no meaning,
# stock_id can be ignored as treating it as categorical take too long to run
# same to date_id
# uncomment for shorter version
# target_lr = ols(formula='target ~  seconds_in_bucket + imbalance_size + C(imbalance_buy_sell_flag) \
#                + reference_price + matched_size + far_price + near_price + bid_price + bid_size \
#                + ask_price + ask_size + wap', data=data)

target_lr = ols(formula='target ~ C(stock_id) + seconds_in_bucket + imbalance_size + C(imbalance_buy_sell_flag) \
                + reference_price + matched_size + far_price + near_price + bid_price + bid_size \
                + ask_price + ask_size + wap', data=data)
target_lr_fit = target_lr.fit()
print("Regression Result for the target")
print(target_lr_fit.summary())
end = time.time()
print(f'The analysis took {(end - start):.1f} seconds')

# %% [markdown]
# Recall target = (wap_t60/wap - index_t60/index) * 10000
#
# Where wap_t60, index_t60 and index are all unknown, hows the correlation looks like
# If we separate them out.
#
# Note we created two new columns,
# * first one for wap_t60/wap * 10000
# * second one for index_t60/index * 10000


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


def correlation_analysis(data, target_column):
    """Perform correlation analysis for the target column and print the results, 

    Args:
        data (dataframe): containing raw data
        target_column (str): dependent variable column
    """
    dependent_column = ['seconds_in_bucket', 'imbalance_size', 'imbalance_buy_sell_flag',
                        'reference_price', 'matched_size', 'far_price', 'near_price',
                        'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap'
                        ]
    pearson_results = pearson_correlation(
        data, target_column, dependent_column)
    spearman_results = spearman_correlation(
        data, target_column, dependent_column)

    sorted_pearson = pd.DataFrame(
        sorted(pearson_results.items(), key=lambda x: abs(x[1]), reverse=True))
    sorted_spearman = pd.DataFrame(
        sorted(spearman_results.items(), key=lambda x: abs(x[1]), reverse=True))

    print("Sorted Pearson Correlation:", sorted_pearson.to_string(index=False))
    print("Sorted Spearman Correlation:",
          sorted_spearman.to_string(index=False))
    return None


data = pd.read_csv(f'{data_path}/train.csv').drop(['row_id'], axis=1)
data.dropna(subset=['target'], inplace=True)
data = process_target(data)
data.drop(['time_id'], axis=1, inplace=True)
data.dropna(inplace=True)

print('\n Analysis for target_wap')
correlation_analysis(data, 'target_wap')

print('\n Analysis for target_index')
correlation_analysis(data, 'target_index')

# %% [markdown]
# Now run the linear regression as baseline
#
# Time_id & row_id has no meaning to lr analysis
#
# Date_id has limited meaning if we dont do time-series
#
# This Regression Analysis will take about 300s
# Due to the stock_id
df = pd.read_csv(
    f'{data_path}/train.csv').drop(['date_id', 'time_id', 'row_id'], axis=1)
df.dropna(subset=['target'], inplace=True)

# print(data.isna().sum())
# columns that have NaN value = imbalance_size, reference_price, matched_size, far_price,
# near_price, bid_price, ask_price, wap, target
start = time.time()
for col in df.columns[df.isnull().any(axis=0)]:
    df[col].fillna(df[col].mean(), inplace=True)

print(df.info())
print(df.head())

df = pd.get_dummies(df, columns=['stock_id'])
features = ['seconds_in_bucket', 'imbalance_size', 'imbalance_buy_sell_flag',
            'reference_price', 'matched_size', 'far_price', 'near_price',
            'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap']

features += [col for col in df.columns if col.startswith(('stock_id_'))]

X = df[features]
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)

end = time.time()
print(f'The analysis took {(end - start):.1f} seconds')
# %%
