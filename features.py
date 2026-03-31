def feature_engineering(df):
    # Example feature: average charge per month
    df['avg_charge'] = df['total_charges'] / (df['tenure'] + 1)

    return df
