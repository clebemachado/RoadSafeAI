import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from model.model_evaluator import ModelEvaluator
from model.tree_model_evaluator import TreeModelEvaluator  # Changed import
from config.inject_logger import inject_logger

@inject_logger
class ModelTrainer:
    """Classe responsável pelo treinamento dos modelos"""
    
    def __init__(self, model: BaseEstimator, name: str):
        self.model = model
        self.name = name
        self.evaluator = ModelEvaluator()
        self.tree_evaluator = TreeModelEvaluator()  # Adiciona o avaliador de árvores
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        self.logger.info(f"Iniciando treinamento do modelo {self.name}...")
        self.model.fit(X_train, y_train)
        self.logger.info(f"Treinamento do modelo {self.name} finalizado")
        return self.model
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "conjunto de dados"
    ) -> Dict:
        self.logger.info(f"Avaliando modelo {self.name} no {dataset_name}...")
        
        y_pred = self.model.predict(X)
        
        try:
            y_prob = self.model.predict_proba(X)
        except AttributeError:
            y_prob = None
        
        metrics = self.evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        self.logger.info(f"Métricas do modelo {self.name} no {dataset_name}:")
        for metric_name, value in metrics.items():
            self.logger.info(f"- {metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Tuple[float, float]:
        self.logger.info(f"Realizando validação cruzada do modelo {self.name}...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        self.logger.info(f"Resultados da validação cruzada ({cv} folds):")
        self.logger.info(f"- Média {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores.mean(), scores.std()
