import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
import scipy.sparse
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv("allstates.csv")
# data['t'] = data['Year'].astype(int) - 2013 # Add 't'to: numericVar and X in order to produce model that accounts for time as a numeric variable.



X = data[['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
          'Political_Control', 'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
          'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax', 'HOV_Access']]
y = data['Evs']

numericVar = ['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
                    'Political_Control', 'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
                    'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax']
cateVar = ['HOV_Access']

# Data pre-processing
numericVarTransformer = Pipeline(steps=[('scaler', StandardScaler())])
categoricalVarTransformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericVarTransformer, numericVar),
        ('cat', categoricalVarTransformer, cateVar)
    ])


X_Processed = preprocessor.fit_transform(X)

if scipy.sparse.issparse(X_Processed):
    X_Processed = X_Processed.toarray()

BRR_model = BayesianRidge()
BRR_model.fit(X_Processed, y)

coefficients_df = pd.DataFrame({
    'Feature': ['Intercept'] + numericVar + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cateVar)),
    'Coefficient': [BRR_model.intercept_] + list(BRR_model.coef_)
})

# Save results to .csv file
coefficients_df.to_csv("name.csv", index=False)


# Performance Metrics

# r^2
y_pred = BRR_model.predict(X_Processed)
rSquared = r2_score(y, y_pred)
print(f'R-squared (Bayesian Regression): {rSquared:.10f}')

# RMSE
RMSE = np.sqrt(mean_squared_error(y, y_pred))
print(f'RMSE: {RMSE:.10f}')