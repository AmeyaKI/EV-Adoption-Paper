# necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# necessary models
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


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
numVarTransformer = Pipeline(steps=[('scaler', StandardScaler())])
cateVarTransformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numVarTransformer, numVar),
        ('cat', cateVarTransformer, cateVar)
    ])

# all 5 models
models = {
    'LinearRegression': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Bayesian Ridge': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', BayesianRidge())
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())
    ]),
    'GBRegressor': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ]),
}

feature_importances = {}
r2_scores = {}
rmse_scores = {}

for model_name, model in models.items():
    model.fit(X, y)
    if model_name in ['LinearRegression', 'Bayesian Ridge']:
        importances = model.named_steps['regressor'].coef_
    elif model_name in ['Random Forest', 'GBRegressor', 'XGBoost']:
        importances = model.named_steps['regressor'].feature_importances_
    feature_names = numVar + list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cateVar))
    
    if len(importances) == len(feature_names):
        feature_importances[model_name] = pd.Series(importances, index=feature_names).abs()
    y_pred = model.predict(X)
    r2_scores[model_name] = r2_score(y, y_pred)
    rmse_scores[model_name] = np.sqrt(mean_squared_error(y, y_pred))


ranks = pd.DataFrame()
for model_name, importances in feature_importances.items():
    ranks[model_name] = importances.rank(ascending=False)

ranks['AverageRank'] = ranks.mean(axis=1)

important_factors = ranks.sort_values('AverageRank').index

# .csv files
ranks.sort_values('AverageRank').to_csv("ensemble_rank_values.csv", index=True)
pd.DataFrame(r2_scores, index=[0]).to_csv("ensemble_r2.csv", index=False)
pd.DataFrame(rmse_scores, index=[0]).to_csv("ensemble_rmse.csv", index=False)