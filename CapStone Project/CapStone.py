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
win = ['2010-01-01', end]
df = pd.DataFrame()
#s&p 500
df['sp500'] = fp.series('SP500').data

#Median Sales Price of Houses Sold for the United States
df['hpr'] = fp.series('MSPUS').data

#Federal Funds Effective Rate
df['fedr'] = fp.series('FEDFUNDS').data

#unemployment rate 
df['unrate'] = fp.series('UNRATE').data

# us gdp
df['gdp'] = fp.series('GDPC1').data

# Equalise the date ranges
#hpr, mhi, cpinf, fedr, sp500 = fp.window_equalize([hpr, mhi, cpinf, fedr, sp500])
df.dropna(inplace = True)
print(df)
df.to_csv('test')

# Create a figure with size (12,12)
fig = plt.figure(figsize=(12, 12))

# First subplot: plot the windowed house prices
ax = fig.add_subplot(5, 1, 1) 
ax.plot(df['hpr'], '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title('hpr')  # Title without the unit part
ax.set_ylabel('hpr')  # Set the unit as labelde here

# second subplot: plot the windowed Fed rate
ax = fig.add_subplot(5, 1, 2) 
ax.plot(df['fedr'], '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title('fedr')  # Title without the unit part
ax.set_ylabel('fdr')  # Set the unit as labelde here

# Third subplot: plot the windowed consumer price inflation
ax = fig.add_subplot(5, 1, 3) 
ax.plot(df['sp500'], '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title('sp500')  # Title without the unit part
ax.set_ylabel('sp500')  # Set the unit as labelde here

# fourth subplot: plot the windowed Real Median Household Income
ax = fig.add_subplot(5, 1, 4) 
ax.plot(df['unrate'], '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title('unrate')  # Title without the unit part
ax.set_ylabel('unrate')  # Set the unit as labelde here

# fourth subplot: plot the windowed Real Median Household Income
ax = fig.add_subplot(5, 1, 5) 
ax.plot(df['gdp'], '-', lw=3, alpha=0.65)
ax.grid()
ax.set_title('gdp')  # Title without the unit part
ax.set_ylabel('gdp')  # Set the unit as labelde here


# Adjust layout to ensure the subplots do not overlap
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)

plt.show()

#computes annual perecatge change (rate of change)
df['hpr_pc'] = df['hpr'].pct_change()
df['fedr_pc'] = df['fedr'].pct_change()
df['sp500_pc'] = df['sp500'].pct_change()
df['unrate_pc'] = df['unrate'].pct_change()
df['gdp_pc']= df['gdp'].pct_change()

# Create a figure with size (12,6) of the percetage changes anually 
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1) 
ax.plot(df['sp500_pc'], linestyle = '-',label = 'sp500', color = 'blue', lw=3, alpha=0.65)
ax.plot(df['hpr_pc'], linestyle ='-',label = 'hpr',color = 'green', lw=3, alpha=0.65)
ax.plot(df['unrate_pc'], linestyle = '-',label = 'unrate_pc', color = 'orange', lw=3, alpha=0.65)
ax.plot(df['gdp_pc'], linestyle ='-',label = 'gdp',color = 'red', lw=3, alpha=0.65)
#ax.plot(df['fedr_pc'], linestyle ='-',label = 'fedr',color = 'purple', lw=3, alpha=0.65)
ax.legend(loc='upper right') #adds legend 
ax.set_title('Comparison of Median Sales Prive vs Median Houshold Income')  # Title 
ax.set_ylabel('%')  # Set the unit as labelde here
ax.set_xlabel("Date")  # Only bottom plot gets the X-axis label
plt.show()

