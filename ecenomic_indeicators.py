import fredpy as fp
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt

# Modify the code below to use your FRED API key:
fp.api_key = 'de118a94573cb3e139ad3fc2f0987e05'
start = '2016-01-01'
end = datetime.now()
## unemployment rate 
unrate = fp.series('UNRATE')
unrate_filtered = unrate.data.loc[start:end]

## GDP
gdp = fp.series('GDP')
gdp_filtered = gdp.data.loc[start:end]

#Consumer Price Index (CPALTT01USM657N)
cpi = fp.series('CPALTT01USM657N')
cpi_filtered =  cpi.data.loc[start:end]

#retail sales 
retail_sales = fp.series('RSXFS')
retail_sales_filtered = retail_sales.data.loc[start:end]

#consumer confidecne index CCI
cci = fp.series('UMCSENT')
cci_filtered = cci.data.loc[start:end]

#s&p 500
sp500 = fp.series('SP500').data.loc[start:end]

fig, axs = plt.subplots(6, figsize=(15, 10))

# Plot UNRATE
axs[0].plot(unrate_filtered.index, unrate_filtered, color='blue', label='UNRATE', lw=3, alpha=0.65)
axs[0].set_ylabel('UNRATE')
axs[0].legend()
axs[0].grid()
# Plot GDP
axs[1].plot(gdp_filtered.index, gdp_filtered, color='green', label='GDP', lw=3, alpha=0.65)
axs[1].set_ylabel('GDP')
axs[1].legend()
axs[1].grid()
# Plot CPI
axs[2].plot(cpi_filtered.index, cpi_filtered, color='red', label='CPI', lw=3, alpha=0.65)
axs[2].set_ylabel('CPI')
axs[2].legend()
axs[2].grid()
# Plot retail sales 
axs[3].plot(retail_sales_filtered.index, retail_sales_filtered, color='brown', label='Retail sales', lw=3, alpha=0.65)
axs[3].set_ylabel('Retail Sales')
axs[3].legend()
axs[3].grid()
# Consumer confidenc index 
axs[4].plot(cci_filtered.index, cci_filtered, color='orange', label='CCI', lw=3, alpha=0.65)
axs[4].set_ylabel('Consumer confidecne Index')
axs[4].legend()
axs[4].grid()
# s&p500
axs[5].plot(sp500.index, sp500, color='yellow', label='S&P500', lw=3, alpha=0.65)
axs[5].set_ylabel('S&P500')
axs[5].legend()
axs[5].grid()

# Adjust layout to ensure the subplots do not overlap
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)

# Set common title and x-axis label
fig.suptitle(f'Filtered Data: UNRATE, GDP Index, CPI,Retail sales, CCI, S&P500 ([{start} to {end})', fontsize=12)
plt.xlabel('Date')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
plt.show()


