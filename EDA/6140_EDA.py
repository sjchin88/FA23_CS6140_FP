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