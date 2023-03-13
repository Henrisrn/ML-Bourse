from tkinter import *
from tkcalendar import DateEntry
import datetime as dt

import pandas_datareader as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if(1==1):
    
    data = web.DataReader("AAPL", 'yahoo')

    # Restructure Data Into OHLC Format
    data = data[['Open', 'High', 'Low', 'Close']]
    print(data)
    # Reset Index And Convert Dates Into Numerical Format
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].map(mdates.date2num)

    # Adjust Style Of The Plot
    ax = plt.subplot()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('{} Share Price'.format("AAPL"), color='white')
    ax.figure.canvas.set_window_title('NeuralNine Stock Visualizer v0.1 Alpha')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis_date()
    