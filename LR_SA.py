import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm
import numpy as np
import scipy.sparse

data = pd.read_csv("allstates.csv")

X = data[['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
          'Political_Control', 'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
          'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax', 'HOV_Access']] 
y = data['Evs']

numVar = ['Infrastructure', 'Electricity_Price', 'Gas_Price', 'Median_Income',
                    'Political_Control', 'Car_Price_Difference', 'Lithium_Price', 'EV_Range',
                    'Combined_EV_Tax_Credit', 'Motor_Fuel_Tax']
cateVar = ['HOV_Access'] 

# data pre-processing
numericVarTransformer = Pipeline(steps=[('scaler', StandardScaler())])
categoricalVarTransformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericVarTransformer, numVar),
        ('cat', categoricalVarTransformer, cateVar)
    ])

X_Processed = preprocessor.fit_transform(X)

if scipy.sparse.issparse(X_Processed):
    X_Processed = X_Processed.toarray()

X_Processed = sm.add_constant(X_Processed)

LR_model = sm.OLS(y, X_Processed).fit()

coefficients = LR_model.params
p_values = LR_model.pvalues

feature_names = ['Intercept'] + numVar
if cateVar:
    feature_names += list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cateVar))

coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'P-value': p_values
})

# .csv file
# coefficients_df.to_csv('name.csv', index=False) - Optional. LR.py measures coefficients/P-values anyways.





# ACTUAL SENSITIVITY ANALYSIS

sensitivityVar = ['Infrastructure', 'Combined_EV_Tax_Credit', 'Gas_Price', 'Electricity_Price', 'Motor_Fuel_Tax']
sensValues = {
    'Infrastructure': np.linspace(0, 1, 11),
    'Combined_EV_Tax_Credit': np.linspace(0, 1, 11),
    'Gas_Price': np.linspace(0, 1, 11),
    'Electricity_Price': np.linspace(0, 1, 11),
    'Motor_Fuel_Tax': np.linspace(0, 1, 11)
}

y_pred_Baseline = LR_model.predict(X_Processed)
deltaAdoption = {var: [] for var in sensitivityVar}
HOV_index = numVar.__len__()




for variable in sensitivityVar:
    for value in sensValues[variable]:
        X_sensitivity = X_Processed.copy()
        
        if variable in numVar:
            idx = numVar.index(variable)
            if scipy.sparse.issparse(X_sensitivity):
                X_sensitivity[:, idx] *= (1 + value)
            else:
                X_sensitivity[:, idx] *= (1 + value)
        elif variable in cateVar:
            X_sensitivity[:, HOV_index] = value
        
        y_pred_sensitivity = LR_model.predict(X_sensitivity)
        
        delta_ev = np.mean(y_pred_sensitivity - y_pred_Baseline)
        deltaAdoption[variable].append(delta_ev)

results = []
for var in sensitivityVar:
    print(f'\nSensitivity Analysis for {var}:')
    for value, deltaCombined in zip(sensValues[var], deltaAdoption[var]):
        print(f'Change in {var} by {value}: Delta EV Adoption (Combined) = {deltaCombined:.6f}')
        results.append({'Variable': var, 'Level': value, 'Delta_EV_Adoption': deltaCombined})




# plotted graph
plt.figure(figsize=(14, 8))

for i, var in enumerate(sensitivityVar):
    plt.plot(sensValues[var], deltaAdoption[var], marker='o', label=var)

plt.xlabel('Change in Sensitivity Variables\n(% Increase)')
plt.ylabel('Delta EV Adoption\n(EVs per 10,000)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()