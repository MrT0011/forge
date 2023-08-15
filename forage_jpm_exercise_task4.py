# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:41:04 2023

@author: Jeffrey
"""


"""
I am not 100% understanding the task. But I tried to optimise r^2 value of the model
via splitting fico_score into n number of categories (quantization)
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm

df = pd.read_csv(r'C:\Users\Jeffrey\Desktop\Github\Task 3 and 4_Loan_Data.csv')


def quantization (fico_score: int, number: int) -> int:
    """
    input:
        fico_score
        number: number of categories to split fico_score into 
    output:
        buckets of fico_score, where the lower the better
    """
    if fico_score > 850:
        print('abnormal score')
        raise Exception()
    
    return number - np.round( fico_score / np.floor(850/number))


def build_model(number: int) -> int:
    """
    input:
        number: number of categories to split fico_score into
    output:
        r^2 of the model
    """
    data = df[['credit_lines_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']].apply(zscore)
    data['fico_score'] = df['fico_score'].apply(lambda x: quantization(x, number))
    data['default'] = df['default']
    data = sm.add_constant(data)
    
    model = sm.Logit(data['default'], data.drop('default',axis=1))

    result = model.fit_regularized()
    
    return result.prsquared

# Dynamic programming for optimisation
result = pd.DataFrame(index=[i for i in range(1, 51)])
prsquared = list()

for i in range(1,51):
    prsquared.append(build_model(i))
    
result['prsquared'] = prsquared
result[result['prsquared'] == result['prsquared'].max()]

"""
Therefore, the best number of categories is 28
"""


