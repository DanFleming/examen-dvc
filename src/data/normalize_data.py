import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

if 'date' in X_train.columns:
    reference_date = pd.to_datetime('2000-01-01')  # Reference date can be anything you want
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_train['date'] = (X_train['date'] - reference_date).dt.days
    
    X_test['date'] = pd.to_datetime(X_test['date'])
    X_test['date'] = (X_test['date'] - reference_date).dt.days

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)
