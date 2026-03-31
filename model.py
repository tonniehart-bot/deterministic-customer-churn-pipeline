import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def split_data(df):
    X = df.drop(columns=['churn', 'customer_id'])
    y = df['churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    np.random.seed(42)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf
