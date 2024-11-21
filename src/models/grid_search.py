import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

if 'date_column' in X_train.columns:
    X_train['date_column'] = pd.to_datetime(X_train['date_column'])
    X_train['year'] = X_train['date_column'].dt.year
    X_train['month'] = X_train['date_column'].dt.month
    X_train['day'] = X_train['date_column'].dt.day
    X_train['weekday'] = X_train['date_column'].dt.weekday
    X_train.drop('date_column', axis=1, inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
param_grid = {'fit_intercept': [True, False]}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

import joblib
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')
