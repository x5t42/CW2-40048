import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
import matplotlib.dates as mdates
import mplfinance as mpf
import openpyxl

df = pd.read_csv('XAUUSD_2010-2023.csv')

df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace = True)
df.sort_index(inplace = True)

print(df.isnull().sum())
df.dropna(inplace = True)

df = df[~df.index.duplicated()]

df['return'] = df['close'].pct_change()
df['volatility'] = df['close'].rolling(window=30).std()
df['rsi_corr'] = df['close'].rolling(window=50).corr(df['rsi14'])

df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['date'] = df.index.date
df['range'] = df['high'] - df['low']

df = df.astype({
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'rsi14': 'float64',
    'sma14': 'float64'
})

# Visualisation 1. Line Plot of Closing Price over Time.

plt.figure(figsize = (16, 6), dpi = 100)
plt.plot(df.index, df['close'], color = 'navy', linewidth = 1.5, label = 'Close Price')

plt.title('Gold Closing Price (2010 - 2023)', fontsize = 18, fontweight = 'bold')
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Price (USD)', fontsize = 14)

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation = 45, fontsize = 10)
plt.yticks(fontsize = 10)

plt.grid(True, linestyle = '--', alpha = 0.5)
plt.legend(loc = 'upper left', fontsize = 12)

plt.tight_layout()
plt.show()

# Visualisation 2. Candlestick Chart.

dfCandle = df[['open', 'high', 'low', 'close']].copy()
dfCandle.columns = ['Open', 'High', 'Low', 'Close']
dfSubset = dfCandle.loc['2010-01-03']

mpf.plot(
    dfSubset,
    type = 'candle',
    style = 'charles',
    title = 'Gold Candlestick Chart (5-Min Intervals)',
    ylabel = 'Price (USD)',
    volume = False,
    figsize = (15, 6),
    savefig = dict(fname = 'gold_candlestick.png', dpi = 300, pad_inches = 0.2)
)

# Visualisation 3. Plot of RSI Over Time.

rsiSubset = df['rsi14'].loc['2010-01-03':'2010-01-10']

plt.figure(figsize = (16, 5))
plt.plot(rsiSubset, color = 'purple', linewidth = 1.5, label = 'RSI (14)')

plt.axhline(70, color = 'red', linestyle = '--', linewidth = 1, label = 'Overbought (70)')
plt.axhline(30, color = 'green', linestyle = '--', linewidth = 1, label = 'Oversold (30)')

plt.title('RSI (14-Period) Over Time', fontsize = 16, fontweight='bold')
plt.xlabel('Date', fontsize = 12)
plt.ylabel('RSI Value', fontsize = 12)
plt.legend(loc = 'upper right')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.tight_layout()
plt.show()

# Visualisation 4. Plot of SMA Over Close Price.

plt.figure(figsize = (16, 5), dpi = 100)

subset = df.loc['2022-03-03':'2022-03-10']

plt.plot(subset['close'], label = 'Close Price', color = 'black', linewidth = 1.5)
plt.plot(subset['sma14'], label = 'SMA (14)', color = 'blue', linestyle = '--', linewidth = 1.5)

plt.title('Gold Close Price vs SMA(14)', fontsize = 16, fontweight = 'bold')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc = 'upper left')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.tight_layout()
plt.show()

# Visualisation 5. Histogram of Returns.

df['return'] = df['close'].pct_change() * 100

df.dropna(subset = ['return'], inplace = True)

plt.figure(figsize=(12, 5), dpi=100)

plt.hist(df['return'], bins = 500, range = (-0.25, 0.25), color = 'steelblue', edgecolor = 'black', alpha = 0.7)

plt.axvline(0, color = 'red', linestyle = '--', linewidth = 1, label = 'Zero Return')
plt.title('Distribution of 5-Minute Returns', fontsize = 16, fontweight = 'bold')
plt.xlabel('Return (%)', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.tight_layout()
plt.show()
