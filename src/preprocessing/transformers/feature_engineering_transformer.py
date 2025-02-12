
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.feature_engineering_03 import FeatureEngineering
from config.inject_logger import inject_logger


@inject_logger
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de engenharia de features"""
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info("Iniciando engenharia de features...")
        return self.feature_engineer.criar_todas_features(X)