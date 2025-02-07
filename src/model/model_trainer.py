import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from model.model_evaluator import ModelEvaluator  # Changed import

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe responsável pelo treinamento dos modelos"""
    
    def __init__(self, model: BaseEstimator, name: str):
        self.model = model
        self.name = name
        self.evaluator = ModelEvaluator()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        logger.info(f"Iniciando treinamento do modelo {self.name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"Treinamento do modelo {self.name} finalizado")
        return self.model
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "conjunto de dados"
    ) -> Dict:
        logger.info(f"Avaliando modelo {self.name} no {dataset_name}...")
        
        y_pred = self.model.predict(X)
        
        try:
            y_prob = self.model.predict_proba(X)
        except AttributeError:
            y_prob = None
        
        metrics = self.evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        logger.info(f"Métricas do modelo {self.name} no {dataset_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"- {metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Tuple[float, float]:
        logger.info(f"Realizando validação cruzada do modelo {self.name}...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        logger.info(f"Resultados da validação cruzada ({cv} folds):")
        logger.info(f"- Média {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores.mean(), scores.std()
