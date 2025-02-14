import pandas as pd

def clean_data(df):
    """Aplica a limpeza de dados removendo colunas irrelevantes, tratando valores nulos e formatando dados."""
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df
