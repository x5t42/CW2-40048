import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('XAUUSD_2010-2023.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

df = df[['open','high','low','sma14','rsi14','close']].dropna()
df['target'] = df['close'].shift(-288)
df.dropna(inplace=True)

split_idx = int(len(df)*0.8)
X = df[['open','high','low','sma14','rsi14']]
y = df['target']

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_test  = y.iloc[split_idx:]

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)
print(f'Linear Regression RMSE: {rmse:.4f}')
print(f'Linear Regression R²:   {r2:.4f}')

results = pd.DataFrame({
    'actual':    y_test,
    'predicted': y_pred
}, index = y_test.index).sort_index()

testStart = results.index[0]
testEnd = testStart + pd.Timedelta(days = 30)
subset = results.loc[testStart:testEnd]

plt.figure(figsize=(12,6))
plt.plot(subset.index, subset['actual'],    label = 'Actual Close',    color = 'black')
plt.plot(subset.index, subset['predicted'], label = 'Predicted Close', color = 'orange', linestyle = 'dashed')
plt.axvline(testStart, color = 'red', linestyle = 'dashed', label = 'Test Start')
plt.title('24-Hours Ahead Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
