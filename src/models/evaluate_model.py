import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

y_test = y_test.squeeze()

model = joblib.load('models/trained_model.pkl')

y_pred = model.predict(X_test_scaled)

y_pred = y_pred.flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {'MSE': mse, 'R2': r2}
import json
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

pd.DataFrame({'true': y_test, 'predicted': y_pred}).to_csv('data/processed_data/predictions.csv', index=False)
