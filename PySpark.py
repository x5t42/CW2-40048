# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:15:55 2025

@author: emily
"""

import sys
import os
print(sys.executable)


#python_path = r"C:\Users\emily\anaconda3\python.exe"
#os.environ["PYSPARK_PYTHON"] = python_path
#os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
#!pip install pyspark


import kagglehub

import pandas as pd
import matplotlib.pyplot as plt


import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_trunc, avg, to_date, lag, when, year, month, day, weekofyear
from pyspark.sql.window import Window


# importing the file using kagglehub
path = kagglehub.dataset_download("ahmadkarrabi/gold-price-archive-2010-2023-dataset")

csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]

gold_prices_csv = os.path.join(path, csv_file)

# begin pyspark session
spark = SparkSession.builder.appName('GoldPrices').getOrCreate()

# import gold prices csv file
gold_prices = (spark.read.format('csv')
               .options(inferSchema = 'true', header = 'true')
               .load(gold_prices_csv))

gold_prices.createOrReplaceTempView('GoldPrices')

# show gold prices table
gold_prices.show()




# show all times where the close value is greater than 1100
gold_prices.filter('close > 1100').show()



# ------------------------- END ----------------------------


# ----------------------------------------------------------------


# PLOT TIME VS CLOSING PRICE ON A GRAPH 

# convert time and close columns to pandas dataframe
gold_pd = gold_prices.select('time', 'close').toPandas()


# visualising closing gold prices over time
gold_pd.set_index('time').plot(figsize=(10,5), title = 'Gold Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# --------------------------- END -------------------------------

# PLOT AVERAGE DAILY RSI IN 2015

# Remove null RSI values
cleaned_rsi = gold_prices.filter(col('rsi14').isNotNull())
cleaned_rsi = cleaned_rsi.withColumn('date', to_date('time'))

# calculate daily RSI average
daily_rsi = (cleaned_rsi
             .groupBy('date')
             .agg(avg('rsi14').alias('rsi14_avg'))
             .orderBy('date'))

# filter the dataset to include just values from 2015
daily_rsi = daily_rsi.filter(year('date') == 2015)

# convert the filtered dataset into a pandas dataframe
rsi_pd = daily_rsi.toPandas()
rsi_pd['date'] = pd.to_datetime(rsi_pd['date'])


# plot the findings against a line for y = 30 and y = 70
plt.plot(rsi_pd['date'], rsi_pd['rsi14_avg'], label = 'Daily Average RSI14', color = 'black')

# y = 70 (overbought gold)
plt.axhline(70, color='green', linestyle='--', linewidth=1.5, label='Overbought (70)')
# y = 30 (undersold gold)
plt.axhline(30, color='red', linestyle='--', linewidth=1.5, label='Oversold (30)')

plt.title('Daily Avg Gold RSI Over Time in 2015')
plt.xlabel('Time')
plt.ylabel('RSI')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------

# CALCULATING DAILY AVERAGE OVER TIME

daily_avg = (gold_prices     
             .withColumn('day', date_trunc('day', col('time'))) # truncate day
             .groupBy('day') # group by day
             .agg(avg('close').alias('avg_close'))   # compute average closing price
             .orderBy('day')) # sort by day

daily_avg.show()


# --------------------------- END -------------------------------


# ---------------------------------------------------------------

# COMPARING AVERAGE GOLD PRICES FOR JAN-MARCH 2018 VS 2023

# find daily average in new dataframe
window = Window.orderBy('day')
daily_avg = daily_avg.withColumn(
    'prev_close', lag('avg_close').over(window))


# compute whether each value is up or down on the previous value
daily_avg = daily_avg.withColumn(
    'direction', when(col('avg_close') > col('prev_close'), 'up')
    .when(col('avg_close') < col('prev_close'), 'down')
    .otherwise('flat'))

# show the daily average dataframe
daily_avg.show()


# filter to find daily average between Jan and March 2018
avg_2018 = daily_avg.filter((year('day') == 2018) &
                                            (month('day').between(1,3)))

# filter to find daily average between Jan and March 2023
avg_2023 = daily_avg.filter((year('day') == 2023) &
                                            (month('day').between(1,3)))


# convert pyspark dataframe to pandas dataframe for plotting
gold_pd_2018 = avg_2018.select('day', 'avg_close', 'direction').toPandas()
gold_pd_2018['day'] = pd.to_datetime(gold_pd_2018['day'])
gold_pd_2018['change'] = gold_pd_2018['avg_close'].diff()
gold_pd_2018.dropna(inplace=True)


# convert pyspark dataframe to pandas dataframe for plotting
gold_pd_2023 = avg_2023.select('day', 'avg_close', 'direction').toPandas()
gold_pd_2023['day'] = pd.to_datetime(gold_pd_2023['day'])
gold_pd_2023['change'] = gold_pd_2023['avg_close'].diff()
gold_pd_2023.dropna(inplace=True)


# find colour of each bar for 2018 and 2023 data
colors_2018 = gold_pd_2018['change'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray'))
colors_2023 = gold_pd_2023['change'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray'))


# plot 2 subplots for 2018 and 2023
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# 2018 subplot
axs[0].bar(gold_pd_2018['day'], gold_pd_2018['change'], color=colors_2018, width=0.8)
axs[0].set_title("Daily Change in Avg Gold Price – Jan–Mar 2018")
axs[0].set_ylabel("Change in Dollars ($)")
axs[0].grid(True)

# 2023 subplot
axs[1].bar(gold_pd_2023['day'], gold_pd_2023['change'], color=colors_2023, width=0.8)
axs[1].set_title("Daily Change in Avg Gold Price – Jan–Mar 2023")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Change in Dollars($)")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------ END -------------------------------
