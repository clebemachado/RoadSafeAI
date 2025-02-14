from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def split_data(df, target, method="stratified", n_splits=5):
    """ Divide os dados para treinamento e teste usando diferentes métodos """
    
    X = df.drop(columns=[target])
    y = df[target]

    if method == "stratified":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(X, y))
    
    elif method == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(kf.split(X, y))
    
    else:
        return train_test_split(X, y, test_size=0.2, random_state=42)
