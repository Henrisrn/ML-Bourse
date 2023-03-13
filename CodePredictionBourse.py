
#%% Import bibli
import yfinance as yf
import pandas as pd

from tkinter import *
from tkcalendar import DateEntry
import datetime as dt

import pandas_datareader as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import datetime as dt
"""
def simple_moving_average(ticker, window):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    sma = data["Close"].rolling(window=window).mean()
    data["SMA"] = sma
    plt.plot(data.index, data["Close"], label="Close")
    plt.plot(data.index, data["SMA"], label="SMA")
    plt.legend()
    plt.show()
    

def linear_regressionn(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    x = np.array(range(len(data))).reshape(-1, 1)
    y = data["Close"].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    plt.plot(data.index, data["Close"], label="Close")
    plt.plot(data.index, y_pred, label="Linear Regression")
    plt.legend()
    plt.show()
    
def linear_regression(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    X = data[["Open", "High", "Low", "Volume"]]
    y = data["Close"]
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    plt.plot(data.index, y, label="Actual")
    plt.plot(data.index, predictions, label="Predicted")
    plt.legend()
    plt.show()

def relative_strength_index(ticker, window):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi
    plt.plot(data.index, data["RSI"])
    plt.axhline(y=30, color='r', linestyle='-')
    plt.axhline(y=70, color='r', linestyle='-')
    plt.show()

def time_series_model(ticker, p, d, q):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    model = ARIMA(data["Close"], order=(p, d, q))
    results = model.fit()
    predictions = results.predict(start="2022-01-01", end="2022-02-27")
    plt.plot(data.index, data["Close"], label="Actual")
    plt.plot(predictions.index, predictions, label="Predicted")
    plt.legend()
    plt.show()

def neural_network_model(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2022-02-27")
    ts = data["Close"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ts.values.reshape(-1,1))
    x_train, y_train = [], []
    for i in range(60,len(scaled_data)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    y_pred = model.predict(x_train)

    # Get the actual Y_test values
    y_test = data["Close"].iloc[-1]

    # Print the results
    print("Predicted value:", scaler.inverse_transform(y_pred)[0][0])
    print("Actual value:", y_test)

    return model

def regression_model(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2022-02-27")
    X = data["Open"].values.reshape(-1, 1)
    y = data["Close"].values
    model = LinearRegression().fit(X, y)
    # Préparation des données de test
    X_test = np.array([100, 105, 110]).reshape(-1, 1)

    # Prédiction avec le modèle
    y_pred = model.predict(X_test)

    # Affichage des prédictions
    print(y_pred)
    return model

def volatility_model(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    returns = np.log(data['Close']/data['Close'].shift(1))
    volatility = np.sqrt(252) * returns.std()
    return volatility


def option_model(ticker, strike, expiry):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    data.fillna(method='ffill', inplace=True)
    S = data['Close'][-1]
    r = 0.01
    expiry = pd.Timestamp(expiry)
    T = (expiry - pd.Timestamp('today')).days / 365
    sigma = data['Close'].pct_change().rolling(20).std().dropna().mean() + 0.001
    d1 = (np.log(S/strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def time_series_modell(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2022-02-27")
    ts = data["Close"]
    model = ARIMA(ts, order=(1,1,1)).fit()
    return model

#%%Import info boite
tickers = pd.read_csv("C://Users//henri//Downloads//nasdaq_screener_1677254480924.csv")
infoboite = pd.DataFrame({
    'Symbol': tickers["Symbol"],
    'Name': tickers["Name"],
    'Country': tickers["Country"],
    'Sector': tickers["Sector"],
    'Industry': tickers["Industry"],    
})
#%%Download des infos sur chaqu'une des boite
print(infoboite)

data = yf.download("AAPL", start="2021-01-01", end="2023-02-24")
cac40 = yf.Ticker("^FCHI")
nasdaq = yf.Ticker("^IXIC")
apple = yf.Ticker("AAPL")
apple_data = apple.history(period="max")
cac40_data = cac40.history(period="max")
nasdaq_data = nasdaq.history(period="max")
# Estimateurs de fonction pour Apple
open_estimator = apple_data['Open'].mean()
high_estimator = apple_data['High'].mean()
low_estimator = apple_data['Low'].mean()
close_estimator = apple_data['Close'].mean()
#adj_close_estimator = apple_data['Adj Close'].mean()
volume_estimator = apple_data['Volume'].mean()

# Tracé des graphiques pour comparer les données d'Apple avec le CAC40 et le Nasdaq 
plt.plot(apple_data['Close'], label='Apple') 
plt.plot(cac40_data['Close'], label='CAC40') 
plt.plot(nasdaq_data['Close'], label='Nasdaq') 
plt.legend() 
#plt.show() 

print("Time Series model :")
print(time_series_modell("AAPL").summary())
print(time_series_modell("AAPL").params)
print(time_series_modell("AAPL").predict())
call_price = option_model("AAPL", 150, pd.to_datetime("2022-01-20"))
print("Call price:", str(call_price))
print("volatility model :"+str(volatility_model("AAPL")))
#print("regression model :")
#print(regression_model("AAPL"))
#print("neural_network model :"+str(neural_network_model("AAPL")))


#simple_moving_average("AAPL", 20)
#linear_regressionn("AAPL")
#relative_strength_index("AAPL", 14)
linear_regression("AAPL", "2021-01-01", "2023-04-27")
#time_series_model("AAPL", 1, 1, 1)



 # Pour connecter Matplotlib avec PowerBI, vous devez d'abord exporter les données sous forme de fichier CSV, puis importer ce fichier CSV dans PowerBI et créer un graphique à partir des données importées.
 
a = pd.DataFrame(data)
print(a)
#a = pd.read_excel("Test.xlsx")
#data = a
Date = pd.DataFrame( a.axes)
df_transpose = Date.transpose()
Date = Date.T[0]

#%% Plot à partir de la donnée

ax = plt.subplot()
ax.grid(True)
ax.set_axisbelow(True)
ax.set_title('{} Share Price'.format("AAPL"), color='white')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis_date()

    # Plot The Candlestick Chart
#candlestick_ohlc(ax, data.values, width=0.5, colorup='#00ff00')
#plt.show()
#%% Analyse Data base
price = data["Close"]
volume = data["Volume"]

print(data.columns.names)
    
#print(price)
#print(volume)
#print(data)

LISTE DES PLOT A FAIRE : 
5 ESTIMATEUR D'UNE FONCTION (REGRESSION LINEAIRE ETC...)
2 PLOT POUR LE COMPARER AVEC LE CAC40
ESSAYER D'EXPORTER CES PLOT SUR POWERBI 
"""




# %% Code issus des différentes vidéo YT

company = 'META'
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)
data = yf.download(company,start,end)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
prediction_days = 60
x_train = []
y_train = []
for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train ,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=25,batch_size=32)
#%% Deuxième partie
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()
test_data = yf.download(company,start,end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)
x_test = []
for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.array(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices,color='black',label=f"Actual {company} Price")
plt.plot(predicted_prices,color='green',label=f"Predicted {company} Price")
plt.title(f"{company} Share price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.show()
