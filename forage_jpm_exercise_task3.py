# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:36:34 2023

@author: Jeffrey
"""

import pandas as pd

df = pd.read_csv(r'C:\Users\Jeffrey\Desktop\Github\Task 3 and 4_Loan_Data.csv')

# data cleaning
# Check if na values
df.isna().sum()
view = df.describe()

"""
Logistic Regression model
1. Choose independent variables and dependent variable
    independent variables:
        customer_id: not relevant
        credit_lines_outstanding: relevant
        loan_amt_outstanding: not using to avoid multi-collinearity with total_debt_outstanding
        total_debt_outstanding: relevant
        income: relevant
        years_employed: relevant
        fico_score: relevant
    
    dependent variable:
        default

2. Prepare data for model
    Normalise each independent variables, using z-score
    
3 Build model

4. Evaluate model
"""

from scipy.stats import zscore
import statsmodels.api as sm
mean = df.mean() # use for building model
sd = df.std() # use for building model

# normalise data
data = df[['credit_lines_outstanding', 'total_debt_outstanding', 'income', 'years_employed']].apply(zscore)
data['default'] = df['default']
data = sm.add_constant(data)

# Build model
model = sm.Logit(data['default'], data.drop('default',axis=1))

result = model.fit_regularized()

# Evaluate model
result.summary2()

# confusion matrix
result.pred_table()

"""
Pseudo R-squared of 0.966
Also, only 68/10000 = 0.68% of data are predicted wrong
"""