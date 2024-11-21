import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

model = LinearRegression(**best_params)
model.fit(X_train_scaled, y_train)

with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
