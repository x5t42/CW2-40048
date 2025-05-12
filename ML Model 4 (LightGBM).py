import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('XAUUSD_2010-2023.csv')

df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace = True)
df.sort_index(inplace = True)

df['target'] = df['close'].shift(-288)

df.dropna(inplace = True)

features = ['open', 'high', 'low', 'sma14', 'rsi14']
X = df[features]
y = df['target']

splitIndex = int(len(df) * 0.8)
xTrain = X.iloc[:splitIndex]
yTrain = y.iloc[:splitIndex]
xTest  = X.iloc[splitIndex:]
yTest  = y.iloc[splitIndex:]

lgbModel = LGBMRegressor(n_estimators = 100, learning_rate = 0.05, max_depth = 5, random_state = 42)
lgbModel.fit(xTrain, yTrain)

yPred = lgbModel.predict(xTest)

rmse = mean_squared_error(yTest, yPred, squared = False)
r2 = r2_score(yTest, yPred)

print(f'LightGBM RMSE: {rmse:.4f}')
print(f'LightGBM R²:   {r2:.4f}')

resultsDf = pd.DataFrame({
    'actual': yTest,
    'predicted': yPred
}, index = yTest.index).sort_index()

testStart = resultsDf.index[0]
testEnd = testStart + pd.Timedelta(days = 30)
subset = resultsDf.loc[testStart:testEnd]

plt.figure(figsize = (12, 6))
plt.plot(subset.index, subset['actual'], label = 'Actual Close', color = 'black')
plt.plot(subset.index, subset['predicted'], label = 'Predicted Close', color = 'orange', linestyle = 'dashed')
plt.axvline(testStart, color = 'red', linestyle = 'dashed', label = 'Test Start')
plt.title('LightGBM: 24-Hours Ahead Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.tight_layout()
plt.show()
