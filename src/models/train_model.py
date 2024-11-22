import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd

best_params = joblib.load('models/best_params.pkl')

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

model = LinearRegression(**best_params)

model.fit(X_train_scaled, y_train)

joblib.dump(model, 'models/trained_model.pkl')
