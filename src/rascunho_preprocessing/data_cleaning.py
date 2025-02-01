import pandas as pd
import numpy as np
import logging
import unidecode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaning:
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros duplicados do dataset."""
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removidos {initial_count - len(df)} registros duplicados.")
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

        initial_count = len(df)
        missing_values = df.isnull().sum().sum()
        logger.info(f"Valores ausentes antes do tratamento: {missing_values}")

        df_treated = df.copy()
        
        only_tipo_acidente_null = df_treated[
            (df_treated['tipo_acidente'].isnull()) & 
            (~df_treated.drop('tipo_acidente', axis=1).isnull().any(axis=1))
        ]
        
        df_treated.loc[only_tipo_acidente_null.index, 'tipo_acidente'] = 'Não informado'
        
        df_treated = df_treated.dropna()
        
        logger.info(f"Linhas onde apenas tipo_acidente era nulo: {len(only_tipo_acidente_null)}")
        logger.info(f"Registros no dataset original: {initial_count}")
        logger.info(f"Registros após tratamento: {len(df_treated)}")
        logger.info(f"Registros removidos: {initial_count - len(df_treated)}")
        
        return df_treated
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza os nomes das colunas para letras minúsculas e sem espaços."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df
    
    @staticmethod
    def normalize_categorical_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Normaliza colunas categóricas removendo acentos e espaços extras."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: unidecode.unidecode(x.strip().lower()) if pd.notnull(x) else x)
        return df
    
    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas as funções de limpeza no dataset."""
        df = DataCleaning.standardize_column_names(df)
        df = DataCleaning.remove_duplicates(df)
        df = DataCleaning.handle_missing_values(df)
        df = DataCleaning.normalize_categorical_columns(df, ['causa_acidente', 'tipo_acidente', 'fase_dia', 'condicao_metereologica'])
        return df
