### 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

data = pd.read_csv('train.csv')
data = data.dropna()


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
    plt.title('Boxplot of '+ var)
    plt.ylabel(var)
    plt.show()
    
###   
    
import matplotlib.pyplot as plt
import seaborn as sns

variables = ['ask_size', 'matched_size', 'imbalance_size', 'far_price', 'near_price', 
             'reference_price', 'bid_price', 'ask_price', 'wap', 'bid_size']

for var in variables:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x=data[var])
    plt.title('Violin plot of ' + var)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()
    
### 
    
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

### 

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

### 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv('train.csv')
df = df.fillna(df.mean())


df = pd.get_dummies(df, columns=['stock_id'])


features = ['seconds_in_bucket', 'imbalance_size', 'imbalance_buy_sell_flag', 
            'reference_price', 'matched_size', 'far_price', 'near_price', 
            'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap']

features += [col for col in df.columns if col.startswith(('stock_id_'))]

X = df[features]
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)