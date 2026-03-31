from preprocess import load_data, clean_data
from features import feature_engineering
from model import split_data, train_models
from evaluate import evaluate

def main():
    # Load & preprocess
    df = load_data('../data/churn_data.csv')
    df = clean_data(df)
    df = feature_engineering(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train
    lr, rf = train_models(X_train, y_train)

    # Evaluate
    lr_acc, lr_roc = evaluate(lr, X_test, y_test)
    rf_acc, rf_roc = evaluate(rf, X_test, y_test)

    print("Logistic Regression -> Accuracy:", lr_acc, "ROC-AUC:", lr_roc)
    print("Random Forest -> Accuracy:", rf_acc, "ROC-AUC:", rf_roc)

if __name__ == "__main__":
    main()
