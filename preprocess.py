import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Handle missing values
    df = df.dropna()

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)

    return df
