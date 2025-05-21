import pandas as pd
import numpy as np
import fredpy as fp
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('classic')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#API key for fred
fp.api_key = 'de118a94573cb3e139ad3fc2f0987e05'

#window to look at data 
end = datetime.now()
win = ['2000-01-01', end]

#Median Sales Price of Houses Sold for the United States
hpr = fp.series('MSPUS')

#Federal Funds Effective Rate
fedr = fp.series('FEDFUNDS')

#Inflation, consumer prices for the United States
cpinf = fp.series('FPCPITOTLZGUSA')

#Real Median Household Income in the United States
mhi = fp.series('MEHOINUSA672N')

#s&p 500
sp500 = fp.series('SP500').window(win)

# Equalise the date ranges
hpr, mhi, cpinf, fedr, sp500 = fp.window_equalize([hpr, mhi, cpinf, fedr, sp500])

# Create a figure with size (12,12)
fig = plt.figure(figsize=(12, 12))

# First subplot: plot the windowed house prices
ax = fig.add_subplot(5, 1, 1) 
ax.plot(hpr.data, '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title(hpr.title.split(':')[0])  # Title without the unit part
ax.set_ylabel(hpr.units)  # Set the unit as labelde here

# second subplot: plot the windowed Fed rate
ax = fig.add_subplot(5, 1, 2) 
ax.plot(fedr.data, '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title(fedr.title.split(':')[0])  # Title without the unit part
ax.set_ylabel(fedr.units)  # Set the unit as labelde here

# Third subplot: plot the windowed consumer price inflation
ax = fig.add_subplot(5, 1, 3) 
ax.plot(cpinf.data, '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title(cpinf.title.split(':')[0])  # Title without the unit part
ax.set_ylabel(cpinf.units)  # Set the unit as labelde here

# fourth subplot: plot the windowed Real Median Household Income
ax = fig.add_subplot(5, 1, 4) 
ax.plot(mhi.data, '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title(mhi.title.split(':')[0])  # Title without the unit part
ax.set_ylabel(mhi.units)  # Set the unit as labelde here

#fith plot: s&p500
ax = fig.add_subplot(5, 1, 5) 
ax.plot(sp500.data, '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title(sp500.title.split(':')[0])  # Title without the unit part
ax.set_ylabel(sp500.units)  # Set the unit as labelde here
ax.set_xlabel("Date")  # Only bottom plot gets the X-axis label

# Adjust layout to ensure the subplots do not overlap
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)

plt.show()

# checks the fquency of the data
print(hpr.frequency)
print(mhi.frequency)
print(cpinf.frequency)
print(fedr.frequency)
print(sp500.frequency)
      
#changes to anual 
hpr_A = hpr.as_frequency(freq='A')
fedr_A = fedr.as_frequency(freq='A')
sp500_A = sp500.as_frequency(freq='A')

#check date range 
sp500_A, cpinf = fp.window_equalize([sp500_A, cpinf])
mhi, hpr_A = fp.window_equalize([mhi, hpr_A])
print('Date range S&P500', sp500_A.date_range)
print('Date range Infation', cpinf.date_range)

#computes annual perecatge change (rate of change)
hpr_A_pi = hpr_A.apc()
mhi_pi = mhi.apc()
fedr_A_pi = fedr_A.apc()
cpinf_pi = cpinf.apc()
sp500_A_pi = sp500_A.apc()

# Create a figure with size (12,6) of the percetage changes anually 
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1) 
ax.plot(mhi_pi.data, linestyle = '-',label = 'mhi', color = 'blue', lw=3, alpha=0.65)
ax.plot(hpr_A_pi.data, linestyle ='-',label = 'hpr',color = 'green', lw=3, alpha=0.65)
ax.legend(loc='upper right') #adds legend 
ax.set_title('Comparison of Median Sales Prive vs Median Houshold Income')  # Title 
ax.set_ylabel(mhi_pi.units)  # Set the unit as labelde here
ax.set_xlabel("Date")  # Only bottom plot gets the X-axis label
plt.show()

# Create a figure with size (12,6) of the percetage changes anually 
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1) 
ax.plot(cpinf_pi.data, linestyle = '-',label = 'Inflation', color = 'blue', lw=3, alpha=0.65)
ax.plot(sp500_A_pi.data, linestyle ='-',label = 'S&P500',color = 'green', lw=3, alpha=0.65)
ax.legend(loc='upper right') #adds legend 
ax.set_title('Comparison of Infation vs S&P500')  # Title 
ax.set_ylabel(sp500_A_pi.units)  # Set the unit as labelde here
ax.set_xlabel("Date")  # Only bottom plot gets the X-axis label
plt.show()