import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran
from preprocessing.merge_datasets import DatasetMerger
from preprocessing.data_cleaning import DataCleaning
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
                
                output_path = f"dados_limpos_{dataset_type}_{result[dataset_type]['data_inversa'].min().year}_{result[dataset_type]['data_inversa'].max().year}.csv"
                result[dataset_type].to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
                logger.info(f"Dataset {dataset_type} limpo salvo em {output_path}")
                
                logger.info(f"Dataset {dataset_type} após limpeza:")
                logger.info(f"Shape: {clean_df.shape}")
                logger.info(f"Valores nulos: {clean_df.isnull().sum().sum()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro durante a limpeza dos datasets: {str(e)}")
            raise

def create_complete_pipeline(collect_new_data: bool = True):
    """Cria o pipeline completo para processamento dos datasets"""
    steps = []
    
    if collect_new_data:
        steps.append(('collect_data', DataCollectionTransformer()))
    
    steps.extend([
        ('merge_datasets', DatasetMergerTransformer()),
        ('clean_data', DataCleaningTransformer())
    ])
    
    return Pipeline(steps)

def main():
    try:
        pipeline = create_complete_pipeline(collect_new_data=False)
        results = pipeline.fit_transform(pd.DataFrame())
        
        for dataset_type, df in results.items():
            logger.info(f"\nResultados finais - Dataset {dataset_type}:")
            logger.info(f"Shape final: {df.shape}")
            logger.info(f"Período: {df['data_inversa'].min()} a {df['data_inversa'].max()}")
            logger.info(f"Total de registros: {len(df)}")
            
            logger.info("\nDistribuição por classificação:")
            logger.info(df['classificacao_acidente'].value_counts())
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()