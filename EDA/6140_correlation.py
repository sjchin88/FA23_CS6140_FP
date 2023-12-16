import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('train.csv')
data.head()

# Assuming 'Close' is the target variable
target = data['target']

# Calculating the range of the target variable
range_min = target.min()
range_max = target.max()
target_range = range_max - range_min

print(f"Range of the target variable: {target_range} (Min: {range_min}, Max: {range_max})")

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

import pandas as pd
from scipy.stats import pearsonr, spearmanr

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

sorted_pearson = sorted(pearson_results.items(), key=lambda x: x[1], reverse=True)
sorted_spearman = sorted(spearman_results.items(), key=lambda x: x[1], reverse=True)

print("Sorted Pearson Correlation:", sorted_pearson)
print("Sorted Spearman Correlation:", sorted_spearman)
