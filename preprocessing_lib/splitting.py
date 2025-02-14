from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.2):
    """Divide os dados em treino e teste mantendo a proporção do target."""
    return train_test_split(df, test_size=test_size, stratify=df[target])
