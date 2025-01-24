import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from typing import Dict, List

class CorrelationAnalysis:
    """Analises de correlação entre variaveis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_numeric_correlations(self) -> pd.DataFrame:
        """Correlações entre variáveis numéricas"""
        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns
        return self.df[numeric_cols].corr().round(2)
    
    def get_categorical_associations(self) -> Dict:
        """Associações entre variáveis categóricas"""
        cat_cols = self.df.select_dtypes(include=['object']).columns
        associations = {}
        
        for col1 in cat_cols:
            for col2 in cat_cols:
                if col1 < col2:
                    contingency = pd.crosstab(self.df[col1], self.df[col2])
                    chi2, p_value = chi2_contingency(contingency)[:2]
                    associations[f"{col1} x {col2}"] = {
                        'chi2': chi2,
                        'p_value': p_value
                    }
        
        return associations
    
    