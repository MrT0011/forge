# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:19:05 2023

@author: Jeffrey
"""

# https://www.theforage.com/modules/bWqaecPDbYAwSDqJy/Ds9LzaAaf5LCF8EjG?ref=NLrbmgCx3mrGFfKs5
import datetime
import statsmodels.api as sm
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np

"""
Read data
"""
df = pd.read_csv(r'C:\Users\Jeffrey\Desktop\Github\Nat_Gas.csv')
df['Dates'] = pd.DatetimeIndex(df['Dates'])
df.set_index('Dates', inplace=True)

# Basic visualisation
"""
Observations
Clear upward trend with seasonality and peak at around January
"""
df.plot(subplots=True)


"""
Regression
Build a model
1. Set dates to to integer (days from 2020-10-31), day 0 = 2020-10-31, day 1 = 2020-11-1 ...
2. Use OLS regression to detrend
3. Use Sinusoidal regression to remove seasonality in the residual of the OLS model
4. Combine both models to forecaste
"""
days = [(i - df.index[0]).days for i in df.index]

# OLS regression
Y = df['Prices']
X = days
X = sm.add_constant(X)
ols_model = sm.OLS(Y,X)
ols_results = ols_model.fit()

ols_res = ols_results.resid
ols_res.index = X[:,1]
# View the residual to ensure the trend is removed
ols_res.plot()

# Sinusoidal regression on OLS residual
# Arbitrary guess for the param
guess_freq = 50
guess_amplitude = np.std(ols_res)**25
guess_phase = 10
guess_offset = np.mean(ols_res)

p0=[guess_freq,
    guess_amplitude,
    guess_phase,
    guess_offset]

def sin_fn(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

# Optimise the parameters
fit = curve_fit(sin_fn, X[:,1], ols_res, p0=p0)

# Fit the values
y_fit = sin_fn(X[:,1], *fit[0])

# View ols_res, fitted values and sin_residual
view = pd.DataFrame({'ols_res':ols_res, 'sin_fitted':y_fit.tolist()})
view['sin_residual'] = view['ols_res'] - view['sin_fitted']

# View the residual and statistics to ensure seasonality is removed
view.plot()
view['sin_residual'].plot()
view.describe()


def forecast(date: datetime):
    """
    This is the final model
    input: any date in the past 
    return one year in the future purchase price of gas
    """
    x = (date - datetime.datetime(2020,10,31)).days
        
    return ols_results.params['const'] + x * ols_results.params['x1'] + sin_fn(x, *fit[0])


"""
Check the final model
"""
model_forecast = [forecast(i) for i in df.index]

model_results = df.copy()
model_results['y_fit'] = model_forecast
model_results['residual'] = model_results['Prices'] - model_results['y_fit']

model_results[['Prices', 'y_fit']].plot(title='Forecast Prices vs Actial Prices')
model_results['residual'].plot(title='Model Residual Plot')
