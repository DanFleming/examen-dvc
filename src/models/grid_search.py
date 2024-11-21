import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

model = LinearRegression()
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

import pickle
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)
