import logging
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TreeModelEvaluator:
    """Classe específica para avaliação de modelos baseados em árvore"""
    
    @staticmethod
    def plot_feature_importance(model: BaseEstimator, feature_names: List[str], top_n: int = 20) -> None:
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importances, x='importance', y='feature')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()