from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.data_cleaning_01 import DataCleaning
from config.inject_logger import inject_logger


@inject_logger
class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de limpeza de dados"""
    def __init__(self):
        self.cleaner = DataCleaning()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info("Iniciando limpeza de dados...")
        return self.cleaner.apply(X)