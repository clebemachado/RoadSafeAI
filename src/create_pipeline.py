import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran
from preprocessing.merge_datasets import DatasetMerger
from preprocessing.data_cleaning import DataCleaning
from preprocessing.feature_engineering import FeatureEngineering
from preprocessing.feature_transformation import FeatureSelection
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollectionTransformer(BaseEstimator, TransformerMixin):
    """Transformador para Coleta de Dados"""
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
    """Transformador para Merge de Datasets"""
    def __init__(self, merger: DatasetMerger = None):
        self.merger = merger or DatasetMerger()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando merge dos datasets...")
        try:
            result = self.merger.execute()
            
            for dataset_type, df in result.items():
                logger.info(f"Dataset {dataset_type}:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Período: {df['data_inversa'].min()} a {df['data_inversa'].max()}")
                logger.info(f"Colunas: {df.columns.tolist()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro durante o merge dos datasets: {str(e)}")
            raise

class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """Transformador para Limpeza de Dados"""
    def __init__(self):
        self.data_cleaning = DataCleaning()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando limpeza dos datasets...")
        try:
            if not isinstance(X, dict) or not all(k in X for k in ['base', 'complete']):
                raise ValueError("Input inválido: esperava um dicionário com chaves 'base' e 'complete'")
            
            result = {}
            
            for dataset_type, df in X.items():
                logger.info(f"Limpando dataset {dataset_type}...")
                clean_df = self.data_cleaning.clean_dataset(df)
                result[dataset_type] = clean_df
                
                logger.info(f"Dataset {dataset_type} após limpeza:")
                logger.info(f"Shape: {clean_df.shape}")
                logger.info(f"Valores nulos: {clean_df.isnull().sum().sum()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro durante a limpeza dos datasets: {str(e)}")
            raise

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Transformador para Feature Engineering"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando feature engineering...")
        try:
            if isinstance(X, dict):
                result = {}
                for dataset_type, df in X.items():
                    logger.info(f"Aplicando feature engineering no dataset {dataset_type}...")
                    result[dataset_type] = FeatureEngineering.apply_feature_engineering(
                        df.copy()
                    )
                return result
            else:
                return FeatureEngineering.apply_feature_engineering(
                    X.copy()
                )
        except Exception as e:
            logger.error(f"Erro durante o feature engineering: {str(e)}")
            raise
        
class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """Transformador para Seleção e Transformação de Features"""
    def __init__(self):
        self.feature_selector = FeatureSelection()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Iniciando seleção e transformação de features...")
        try:
            if isinstance(X, dict):
                result = {}
                for dataset_type, df in X.items():
                    logger.info(f"Aplicando transformações no dataset {dataset_type}...")
                    result[dataset_type] = self.feature_selector.transform_features(df.copy())
                return result
            else:
                return self.feature_selector.transform_features(X.copy())
                
        except Exception as e:
            logger.error(f"Erro durante a seleção e transformação de features: {str(e)}")
            raise

def save_processed_datasets(datasets: dict):
    """Função para salvar os datasets processados"""
    try:
        for dataset_type, df in datasets.items():
            output_path = f"dados_processados_{dataset_type}_{df['data_inversa'].min().year}_{df['data_inversa'].max().year}.csv"
            df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
            logger.info(f"Dataset {dataset_type} processado salvo em {output_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar os datasets: {str(e)}")
        raise

def create_complete_pipeline(collect_new_data: bool = True):
    """Cria o pipeline completo para processamento dos datasets"""
    steps = []
    
    if collect_new_data:
        steps.append(('collect_data', DataCollectionTransformer()))
    
    steps.extend([
        ('merge_datasets', DatasetMergerTransformer()),
        ('clean_data', DataCleaningTransformer()),
        ('feature_engineering', FeatureEngineeringTransformer())
    ])
    
    return Pipeline(steps)

def main():
    try:
        pipeline = create_complete_pipeline(collect_new_data=False)
        results = pipeline.fit_transform(pd.DataFrame())
        
        save_processed_datasets(results)
        
        for dataset_type, df in results.items():
            logger.info(f"\nResultados finais - Dataset {dataset_type}:")
            logger.info(f"Shape final: {df.shape}")
            logger.info(f"Período: {df['data_inversa'].min()} a {df['data_inversa'].max()}")
            logger.info(f"Total de registros: {len(df)}")
            
            logger.info("\nDistribuição por classificação:")
            logger.info(df['classificacao_acidente'].value_counts())
            
            if 'severidade' in df.columns:
                logger.info("\nDistribuição por severidade:")
                logger.info(df['severidade'].value_counts())
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()