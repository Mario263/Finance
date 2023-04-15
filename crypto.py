##make 4 copies of this app for crypto,stock ,mf ,gold
##deployment of app
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from keras.models import load_model
import streamlit as st 

import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.title('CRYPTO TREND ANALYSIS')
user_input = st.text_input('Enter Crypto Ticker' , 'BTC-USD')

df_ticker = yf.Ticker(user_input)
df = df_ticker.history(period="max", auto_adjust=True) 

## 1 year , 3 years ,5 months dataframe

df3year = df.iloc[-1095:]
df1year = df.iloc[-365:]
df5month = df.iloc[-153:]


##CANDLESTICK CHART

st.subheader('CANDLESTICK CHART')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = go.Figure()
fig = make_subplots(rows=1, cols=1)
fig.append_trace(go.Candlestick(x=df1year.index,open=df1year.Open,high=df1year.High,low=df1year.Low,close=df1year.Close,  ), row=1, col=1)
fig.update_layout(xaxis_title="Date",yaxis_title="Price",width=1500,height=600,template = "plotly_dark")
st.plotly_chart(fig)

#Describing Data

st.subheader('SUMMARY')
st.write(df.tail(5))
st.write(df.describe())

#Closed Price

st.subheader('CLOSED PRICE CHART')
fig = plt.figure(figsize = (12,8))
ax1 = plt.subplot(211)
ax1.plot(df.index, df["Close"], color = "lightgrey")
ax1.set_title("Close Price", color="white", fontweight="bold")
ax1.set_ylabel("Price", color="white")
ax1.grid(True, color="#555555")
ax1.set_axisbelow(True)
ax1.set_facecolor('black') #colour of individual plot
ax1.figure.set_facecolor('#121212') # colour of full subplot
ax1.tick_params(axis ='x', colors='white')
ax1.tick_params(axis ='y', colors='white')
st.pyplot(fig)


#Visualisation with 50 & 100 MA

st.subheader('INDICATORS USED')
st.subheader('1. 50 & 100 DAYS SIMPLE MOVING AVERAGE CHART (3YRS)')
ma50 = df3year.Close.rolling(50).mean()
ma100 = df3year.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot(211)
ax1.plot(df3year.index, df3year["Close"], color = "lightgrey")
ax1.plot(df3year.index, ma50, color = "green")
ax1.plot(df3year.index, ma100, color = "orange")
ax1.set_title("Close Price of Last 3 Years", color = "white")
ax1.set_ylabel("Price in USD", color = "white")
ax1.grid(True, color = "#555555")
ax1.set_axisbelow(True)
ax1.set_facecolor("black")
ax1.figure.set_facecolor("#121212")
ax1.tick_params(axis = "x", colors ="white") #notice here 'colors' is used insread of 'color' 
ax1.tick_params(axis = "y", colors ="white")
plt.legend(["Close Price","50 Day Moving Average","100 Day Moving Average"], facecolor="black", labelcolor = "white")
st.pyplot(fig)

#Comparison between 50 MA & 50 EMA

st.subheader('2. 50 DAYS EXPONENTIAL MOVING AVERAGE CHART (3YRS)')
#Calculating Exponential Moving Average
ema50 = df3year.Close.ewm(span=50,min_periods=50).mean()

#Plotting Simple Moving Average and Weighted moving Average
fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot(211)
ax1.plot(df3year.index, df3year["Close"], color = "lightgrey")
ax1.plot(df3year.index, ma50, color = "orange")
ax1.plot(df3year.index, ema50, color = "red")
ax1.set_title("Close Price of Last 3 Years", color = "white")
ax1.set_ylabel("Price in USD", color = "white")
ax1.grid(True, color = "#555555")
ax1.set_axisbelow(True)
ax1.set_facecolor("black")
ax1.figure.set_facecolor("#121212")
ax1.tick_params(axis = "x", colors ="white") #notice here 'colors' is used insread of 'color' 
ax1.tick_params(axis = "y", colors ="white")
plt.legend(["Close Price","50 Day Moving Average","50 Day Exponential Moving Average"], facecolor="black", labelcolor = "white")
st.pyplot(fig)

# RELATIVE STRENGTH INDEX (WILDER'S RSI) for 3 years

delta = df['Close'].diff()

delta.dropna(inplace=True)

positive = delta.copy()
positive[positive < 0] = 0

negative = delta.copy()
negative[negative > 0] = 0

#Wilder recommended a smoothing period of 14 days
days=14

average_gain = positive.rolling(window = days).mean()
average_loss = abs(negative.rolling(window = days).mean())


relative_strength = average_gain / average_loss
RSI = 100.0 - (100.0 / (1.0 + relative_strength))

df1 = df.copy()
df1['RSI'] = RSI
df3 = df1.iloc[-1095:]

st.subheader('3. RELATIVE STRENGTH INDEX CHART (3YRS)')

fig = plt.figure(figsize = (12,9))
ax1 = plt.subplot(211)
ax1.plot(df3.index, df3['Close'], color='lightgrey')
ax1.set_title("Close Price",color='white')

ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis ='x', colors='white')
ax1.tick_params(axis ='y', colors='white')

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(df3.index, df3['RSI'],color='lightgrey')
ax2.axhline(1, linestyle='--', alpha=0.5, color='#ff0000')
ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')

ax2.set_title("RSI Value",color='white')
ax2.grid(False)
ax2.set_axisbelow(True)
ax2.set_facecolor('black')
ax2.tick_params(axis ='x', colors='white')
ax2.tick_params(axis ='y', colors='white')
st.pyplot(fig)

#MOVING AVERAGE CONVERGENCE DIVERGENCE

ema1 = df.Close.ewm(span = 12).mean()
ema2 = df.Close.ewm(span = 26).mean()

macd = ema1 - ema2
signal = macd.ewm(span = 9).mean()


df1year = df.iloc[-365:]
macd1year = macd.iloc[-365:]
signal1year = signal.iloc[-365:]

st.subheader('4. MOVING AVERAGE CONVERGENCE DIVERGENCE CHART (1YR)')
fig = plt.figure(figsize = (12,9))
ax1 = plt.subplot(211)
ax1.plot(df1year.index, df1year['Close'], color='lightgrey')
ax1.set_title("Close Price",color='white')

ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis ='x', colors='white')
ax1.tick_params(axis ='y', colors='white')



ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(df1year.index, signal1year, color='yellow')
ax2.plot(df1year.index, macd1year, color='lightgrey')
ax2.plot(df1year.index, signal1year, color='yellow')
ax2.set_title("Close Price",color='white')

ax2.grid(True, color='#555555')
ax2.set_axisbelow(True)
ax2.set_facecolor('black')
ax2.figure.set_facecolor('#121212')
ax2.tick_params(axis ='x', colors='white')
ax2.tick_params(axis ='y', colors='white')
st.pyplot(fig)

#Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load the model

model = load_model('keras_model.h5')


#Testing

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing])
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
  x_test.append(input_data[i-100 : i])
  y_test.append(input_data[i , 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph

st.subheader('PREDICTED vs ORIGINAL CLOSING PRICE CHART')
fig2 = plt.figure(figsize=(12,6))
ax1 = plt.subplot(211)
ax1.plot(y_test , 'b' , label = 'Original Price')
ax1.plot(y_predicted , 'r' , label = 'Predicted Price')

ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis ='x', colors='white')
ax1.tick_params(axis ='y', colors='white')

ax1.legend(["Original Price","Predicted Price"],facecolor="black", labelcolor = "white")
st.pyplot(fig2)