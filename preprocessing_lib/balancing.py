from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    """Aplica a técnica SMOTE para balancear o dataset."""
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
