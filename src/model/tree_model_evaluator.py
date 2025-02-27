import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from config.inject_logger import inject_logger

@inject_logger
class TreeModelEvaluator:
    """Classe específica para avaliação de modelos baseados em árvore"""
    
    def plot_feature_importance(self, model: BaseEstimator, feature_names: List[str], top_n: int = 20) -> None:
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False).head(top_n)
        
        sns.barplot(data=importances, x='importance', y='feature')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.tight_layout()