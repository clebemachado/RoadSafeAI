from imblearn.over_sampling import SMOTE, ADASYN

def balance_data(X, y, method="smote"):
    """ Aplica balanceamento de classes no dataset """
    
    if method == "smote":
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    
    elif method == "adasyn":
        adasyn = ADASYN(sampling_strategy="auto", random_state=42)
        X_res, y_res = adasyn.fit_resample(X, y)
    
    return X_res, y_res
