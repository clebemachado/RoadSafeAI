from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.data_standardize_02 import DataStandardize
from config.inject_logger import inject_logger

@inject_logger
class DataStandardizeTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de padronização de dados"""
    def __init__(self):
        self.standardizer = DataStandardize()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info("Iniciando padronização de dados...")
        return self.standardizer.padronizar_dataset(X)