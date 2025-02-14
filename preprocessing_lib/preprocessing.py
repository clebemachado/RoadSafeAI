def preprocess_data(df):
    """Aplica transformações como normalização, padronização e criação de features."""
    # Exemplo: Convertendo datas para datetime
    df["data_inversa"] = pd.to_datetime(df["data_inversa"])
    return df
