import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import scipy.sparse
from sklearn.metrics import mean_squared_error
import numpy as np


# Data pre-porocessing
data = pd.read_csv("allstates.csv")
data['t'] = data['Year'].astype(int) - 2013 # Add 't'to: numericVar and X in order to produce model that accounts for time as a numeric variable.


X = data[['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
            'Political_Control', 'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
            'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax', 'HOV_Access']] 
y = data['Evs']

numericVar = ['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
                     'Political_Control',  'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
                    'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax']
categoricalVar = ['HOV_Access'] 

numericVarTransformer = Pipeline(steps=[('scaler', StandardScaler())])
categoricalVarTransformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericVarTransformer, numericVar),
        ('cat', categoricalVarTransformer, categoricalVar)
    ])

X_Processed = preprocessor.fit_transform(X)

if scipy.sparse.issparse(X_Processed):
    X_Processed = X_Processed.toarray()

X_Processed = sm.add_constant(X_Processed)

LR_model = sm.OLS(y, X_Processed).fit()

# Calculating coefficients and p-values
coefficients = LR_model.params
p_values = LR_model.pvalues

feature_names = numericVar + ['Intercept']
if categoricalVar:
    feature_names += list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categoricalVar))

coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'P-value': p_values
})


# Results saved to .csv file
coefficients_df.to_csv("name.csv", index=False)



# Performance Metrics

# r^2
r2 = r2_score(y, LR_model.predict(X_Processed))
print(f'R-squared: {r2:.10f}')

# RMSE
y_pred = LR_model.predict(X_Processed)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'RMSE: {rmse:.10f}')