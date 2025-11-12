
''' This module contains functions for data preprocessing, including cleaning,
    transforming, and balancing the dataset for modeling purposes. '''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def preprocessing(df):
    ''' Cleans and transforms raw data into processed data, applying mapping
        for binary variables, one-hot encoding for some categorical variables
        and returns X, Y for data modeling afterwards  . '''
    
    rd = df.copy()
    mappings = {
    'attrition_flag': {'Existing Customer': 0, 'Attrited Customer': 1},
    'gender': {'F': 0, 'M': 1}
    }

    for col, mapping in mappings.items():
        rd[col] = rd[col].map(mapping)

    rd = rd.drop(columns = ['clientnum'], axis=1)
    non_numeric_cols = rd.select_dtypes(include=['object']).columns
    rd = pd.get_dummies(rd, columns=non_numeric_cols, drop_first=True)

    X, Y = rd.drop('attrition_flag', axis=1), rd['attrition_flag']
    print("Preprocessing completed: Data cleaned and transformed. \n")
    print(f"Y proportion: \n{Y.value_counts(normalize=True)} \n")
    return X, Y

def balance_data(X, Y, sampling_strategy, method='oversample'):
    ''' Balances the dataset using the specified method: 'oversample', 'undersample', or 'smotetomek'.
        Returns the balanced X and Y datasets. By default, it uses random oversampling. '''

    if method == 'oversample':
        sampler = RandomOverSampler(random_state=472, sampling_strategy=sampling_strategy)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=472, sampling_strategy=sampling_strategy)
    elif method == 'smotetomek':
        sampler = SMOTETomek(random_state=472, sampling_strategy=sampling_strategy)
    else:
        raise ValueError("Method must be 'oversample', 'undersample', or 'smotetomek'.")
# Restaurar nombres y tipo
    X_resampled, Y_resampled = sampler.fit_resample(X, Y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    Y_resampled = pd.Series(Y_resampled, name=Y.name)
    print(f"Data balancing completed: Applied {method} with sampling strategy {sampling_strategy}. \n")
    print(f"Y_balanced proportion: \n {Y_resampled.value_counts(normalize=True)}")
    return X_resampled, Y_resampled


# -----------------Example usage: --------------

##df = pd.read_excel(r"C:\Users\LENOVO\Documents\D\Documentos\Javeriana\IX semestre\Analítica 2\Proyecto_Final\data\bank_churn.xlsx")
##print("Reading data completed. \n")
##print("Starting preprocessing... \n")
##x, y = preprocessing(df)
##print("Preprocessing completed. \n")
##x_balanced, y_balanced = balance_data(x, y, sampling_strategy=0.5, method='oversample')
##print("Data balancing completed. \n")
##print(y_balanced.value_counts(normalize=True))
