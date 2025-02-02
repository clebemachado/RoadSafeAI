import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Dict, List

class FeatureAnalysis:
    """Análise das features do dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def get_missing_values(self) -> pd.DataFrame:
        """Retorna a contagem e a porcentagem de valores ausentes em cada coluna."""
        missing = self.df.isnull().sum()
        missing_percentage = (missing / len(self.df)) * 100
        return pd.DataFrame({'Valores Ausentes': missing, 'Porcentagem': missing_percentage}).sort_values(by='Valores Ausentes', ascending=False)
    
    def get_feature_distribution(self, feature: str) -> pd.DataFrame:
        """Retorna a distribuição de valores de uma feature específica."""
        if feature in self.df.columns:
            return self.df[feature].value_counts(normalize=True).round(4) * 100
        else:
            raise ValueError(f"A feature '{feature}' não existe no dataset.")
    
    def get_feature_importance(self, target: str, n_features: int = 10) -> pd.DataFrame:
        """Retorna a importância das features usando um Random Forest Classifier."""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(self.df.dropna(), columns=categorical_cols, drop_first=True)
        
        if target not in df_encoded.columns:
            raise ValueError(f"A variável target '{target}' não está no dataset.")
        
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_imputed, y)
        
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importância': model.feature_importances_})
        return feature_importances.sort_values(by='Importância', ascending=False).head(n_features).reset_index(drop=True)