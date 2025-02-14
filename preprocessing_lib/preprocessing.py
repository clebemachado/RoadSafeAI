import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """ Tratamento de valores ausentes e pré-processamento básico """

    numeric_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Pipeline para imputação de valores numéricos e categóricos
    imputer_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))  # Nova imputação avançada
    ])

    df[numeric_features] = imputer_pipeline.fit_transform(df[numeric_features])
    df[categorical_features] = df[categorical_features].fillna("Desconhecido")

    return df
