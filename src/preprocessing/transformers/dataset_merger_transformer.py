
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from config.inject_logger import inject_logger
from data_collection.merge_datasets import DatasetMerger


@inject_logger
class DatasetMergerTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de união dos datasets"""
    def __init__(self, merger: DatasetMerger = None, dataset_type: Literal['base', 'complete'] = 'base'):
        """
        Inicializa o transformador de união de datasets.
        """
        self.merger = merger or DatasetMerger()
        self.dataset_type = dataset_type
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info(f"Iniciando união dos datasets (usando dataset {self.dataset_type})...")
        try:
            result = self.merger.execute()
            
            if self.dataset_type not in result:
                raise ValueError(f"Tipo de dataset '{self.dataset_type}' não encontrado nos resultados")
            
            selected_dataset = result[self.dataset_type]
            self.logger.info(f"Dataset {self.dataset_type} selecionado:")
            self.logger.info(f"Dimensões: {selected_dataset.shape}")
            self.logger.info(f"Período: {selected_dataset['data_inversa'].min()} até {selected_dataset['data_inversa'].max()}")
            self.logger.info(f"Colunas: {selected_dataset.columns.tolist()}")
            
            return selected_dataset
            
        except Exception as e:
            self.logger.error(f"Erro durante a união dos datasets: {str(e)}")
            raise