import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical(df, categorical_columns):
    """ Aplica encoding One-Hot em variáveis categóricas """
    
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_columns])
    
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, pd.DataFrame(encoded_cols)], axis=1)
    
    return df
