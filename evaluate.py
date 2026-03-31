from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, probs)

    return acc, roc
