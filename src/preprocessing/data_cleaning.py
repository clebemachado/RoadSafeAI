import pandas as pd
import numpy as np
import logging

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
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Trata valores ausentes no dataset."""
        missing_values = df.isnull().sum().sum()
        logger.info(f"Valores ausentes antes do tratamento: {missing_values}")
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'median':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        missing_values_after = df.isnull().sum().sum()
        logger.info(f"Valores ausentes após o tratamento: {missing_values_after}")
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: list, threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers com base no método do IQR (Interquartile Range)."""
        initial_count = len(df)
        for col in columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (IQR * threshold)
                upper_bound = Q3 + (IQR * threshold)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Removidos {initial_count - len(df)} outliers das colunas {columns}.")
        return df
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza os nomes das colunas para letras minúsculas e sem espaços."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df
    
    @staticmethod
    def normalize_categorical_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Normaliza colunas categóricas removendo acentos e espaços extras."""
        import unidecode
        
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
        df = DataCleaning.remove_outliers(df, ['km', 'pessoas', 'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'veiculos'])
        df = DataCleaning.normalize_categorical_columns(df, ['causa_acidente', 'tipo_acidente', 'fase_dia', 'condicao_metereologica'])
        return df
