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