from sklearn.base import BaseEstimator, TransformerMixin
from config.inject_logger import inject_logger
from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran

import pandas as pd

@inject_logger
class DataCollectionTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de coleta de dados"""
    def __init__(self, collector: CollectData = None):
        self.collector = collector or CollectDataDetran()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info("Iniciando coleta de dados...")
        try:
            self.collector.execute()
            self.logger.info("Coleta de dados concluída com sucesso")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Erro durante a coleta de dados: {str(e)}")
            raise