from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical(df, column, method="label"):
    """Codifica variáveis categóricas usando Label Encoding ou One-Hot Encoding."""
    if method == "label":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
    elif method == "onehot":
        df = pd.get_dummies(df, columns=[column])
    return df
