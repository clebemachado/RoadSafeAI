import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran
from data_collection.merge_datasets import DatasetMerger
from preprocessing.data_balancing_06 import DataBalance
from preprocessing.data_cleaning_01 import DataCleaning
from preprocessing.data_encoding_04 import DataEncoding
from preprocessing.data_split_05 import DataSplit
from preprocessing.data_standardize_02 import DataStandardize
from preprocessing.feature_engineering_03 import FeatureEngineering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollectionTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de coleta de dados"""
    def __init__(self, collector: CollectData = None):
        self.collector = collector or CollectDataDetran()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando coleta de dados...")
        try:
            self.collector.execute()
            logger.info("Coleta de dados concluída com sucesso")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro durante a coleta de dados: {str(e)}")
            raise

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
        logger.info(f"Iniciando união dos datasets (usando dataset {self.dataset_type})...")
        try:
            result = self.merger.execute()
            
            if self.dataset_type not in result:
                raise ValueError(f"Tipo de dataset '{self.dataset_type}' não encontrado nos resultados")
            
            selected_dataset = result[self.dataset_type]
            logger.info(f"Dataset {self.dataset_type} selecionado:")
            logger.info(f"Dimensões: {selected_dataset.shape}")
            logger.info(f"Período: {selected_dataset['data_inversa'].min()} até {selected_dataset['data_inversa'].max()}")
            logger.info(f"Colunas: {selected_dataset.columns.tolist()}")
            
            return selected_dataset
            
        except Exception as e:
            logger.error(f"Erro durante a união dos datasets: {str(e)}")
            raise

class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de limpeza de dados"""
    def __init__(self):
        self.cleaner = DataCleaning()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando limpeza de dados...")
        return self.cleaner.clean_dataset(X)

class DataStandardizeTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de padronização de dados"""
    def __init__(self):
        self.standardizer = DataStandardize()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando padronização de dados...")
        return self.standardizer.padronizar_dataset(X)

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de engenharia de features"""
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando engenharia de features...")
        return self.feature_engineer.criar_todas_features(X)

class DataEncodingTransformer(BaseEstimator, TransformerMixin):
    """Transformador para etapa de codificação de dados"""
    def __init__(self):
        self.encoder = DataEncoding()
        self.fitted = False
    
    def fit(self, X, y=None):
        if not self.fitted:
            logger.info("Ajustando codificador de dados...")
            self.encoder.fit(X, 'gravidade_acidente')
            self.fitted = True
        return self
    
    def transform(self, X):
        logger.info("Iniciando codificação de dados...")
        return self.encoder.transform(X, 'gravidade_acidente')

class PreprocessingPipeline:
    """
    Classe principal responsável por gerenciar o pipeline completo de pré-processamento,
    desde a coleta de dados até o pré-processamento final
    """
    def __init__(
        self,
        collect_new_data: bool = True,
        dataset_type: Literal['base', 'complete'] = 'base',
        test_size: float = 0.2,
        valid_size: float = 0.2,
        balance_strategy: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Inicializa o pipeline de pré-processamento
        """
        self.collect_new_data = collect_new_data
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.valid_size = valid_size
        self.balance_strategy = balance_strategy
        self.random_state = random_state
        
        steps = []
        
        if self.collect_new_data:
            steps.extend([
                ('collect_data', DataCollectionTransformer()),
                ('merge_datasets', DatasetMergerTransformer(dataset_type=self.dataset_type))
            ])
        
        steps.extend([
            ('cleaning', DataCleaningTransformer()),
            ('standardize', DataStandardizeTransformer()),
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('encoding', DataEncodingTransformer())
        ])
        
        self.pipeline = Pipeline(steps)
        
        self.data_splitter = DataSplit()
        
        self.data_balancer = DataBalance() if balance_strategy else None

    def process_data(
        self,
        input_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Processa os dados através do pipeline completo
        """
        logger.info(f"Iniciando pipeline completo de pré-processamento (usando dataset {self.dataset_type})...")
        
        df_processed = self.pipeline.fit_transform(input_data if input_data is not None else pd.DataFrame())
        logger.info(f"Dimensões após pré-processamento: {df_processed.shape}")
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_splitter.prepare_data(
            df_processed,
            test_size=self.test_size,
            valid_size=self.valid_size,
            random_state=self.random_state
        )
        
        # Garantir que todos os tipos numéricos sejam float64 antes do balanceamento
        numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[numeric_columns] = X_train[numeric_columns].astype('float64')
        X_valid[numeric_columns] = X_valid[numeric_columns].astype('float64')
        X_test[numeric_columns] = X_test[numeric_columns].astype('float64')
        
        logger.info("Tipos de dados após conversão:")
        for col in X_train.columns:
            logger.info(f"- {col}: {X_train[col].dtype}")
        
        if self.balance_strategy:
            logger.info(f"Aplicando estratégia de balanceamento {self.balance_strategy}...")
            try:
                X_train, y_train = self.data_balancer.balance_data(
                    X_train,
                    y_train,
                    strategy=self.balance_strategy,
                    random_state=self.random_state
                )
            except Exception as e:
                logger.error(f"Erro durante o balanceamento: {str(e)}")
                logger.error("Tipos de dados após tentativa de balanceamento:")
                for col in X_train.columns:
                    logger.error(f"- {col}: {X_train[col].dtype}")
                raise
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def get_feature_names(self) -> Dict[str, list]:
        """
        Obtém os nomes das features após pré-processamento
        """
        encoder = self.pipeline.named_steps['encoding'].encoder
        return encoder.get_feature_names()

def main():
    """Exemplo de uso do pipeline"""
    try:
        pipeline = PreprocessingPipeline(
            collect_new_data=True,
            dataset_type='base',
            test_size=0.2,
            valid_size=0.2,
            balance_strategy='smote',
            random_state=42
        )
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = pipeline.process_data()
        
        feature_names = pipeline.get_feature_names()
        
        logger.info("\nPré-processamento finalizado com sucesso!")
        logger.info(f"Formato dados de treino: {X_train.shape}")
        logger.info(f"Formato dados de validação: {X_valid.shape}")
        logger.info(f"Formato dados de teste: {X_test.shape}")
        logger.info("\nFeatures:")
        for feature_type, features in feature_names.items():
            logger.info(f"- {feature_type}: {len(features)} features")
        
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()