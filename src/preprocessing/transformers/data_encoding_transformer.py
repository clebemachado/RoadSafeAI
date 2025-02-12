
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.data_encoding_04 import DataEncoding
from config.inject_logger import inject_logger

@inject_logger
class DataEncodingTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de codificação de dados"""
    def __init__(self):
        self.encoder = DataEncoding()
        self.fitted = False
    
    def fit(self, X, y=None):
        if not self.fitted:
            self.logger.info("Ajustando codificador de dados...")
            self.encoder.fit(X, 'gravidade_acidente')
            self.fitted = True
        return self
    
    def transform(self, X):
        self.logger.info("Iniciando codificação de dados...")
        return self.encoder.transform(X, 'gravidade_acidente')