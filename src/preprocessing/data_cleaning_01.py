import pandas as pd
import numpy as np
import logging
import unidecode
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaning:
    """
    Classe responsável pela limpeza e preparação do dataset de acidentes.
    """
    
    # Colunas para remover
    COLUMNS_TO_DROP = ['id', 'unnamed: 0', 'uf', 'tracado_via', 'feridos','fase_dia']

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros duplicados do dataset."""
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        logger.info(f"Removidos {duplicates_removed} registros duplicados.")
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas que contêm valores nulos, vazios ou variações de null do dataframe.
        """
        df_clean = df.copy()
        
        null_values = ['null', '(null)', 'NULL', '(NULL)', 'NaN', 'nan', 'NAN', 
                    'undefined', '', ' ', None]
        
        linhas_inicial = len(df_clean)
        
        df_clean = df_clean.replace(null_values, np.nan)
        
        def contem_apenas_espacos(x):
            return isinstance(x, str) and x.isspace()
        
        mascara_espacos = pd.DataFrame(False, index=df_clean.index, columns=['remove'])
        for coluna in df_clean.select_dtypes(include=['object']).columns:
            mascara_espacos['remove'] |= df_clean[coluna].apply(contem_apenas_espacos)
        
        df_clean = df_clean.dropna()
        df_clean = df_clean[~mascara_espacos['remove']]
        
        linhas_removidas = linhas_inicial - len(df_clean)
        percentual_removido = (linhas_removidas / linhas_inicial) * 100
        
        logger.info(f"Estatísticas de limpeza:")
        logger.info(f"- Linhas no dataset original: {linhas_inicial}")
        logger.info(f"- Linhas removidas: {linhas_removidas}")
        logger.info(f"- Percentual removido: {percentual_removido:.4f}%")
        logger.info(f"- Linhas no dataset final: {len(df_clean)}")
        
        return df_clean

    @staticmethod
    def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas irrelevantes do dataset."""
        columns_to_drop = [col for col in DataCleaning.COLUMNS_TO_DROP if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Colunas removidas: {columns_to_drop}")
        return df

    @classmethod
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as etapas de limpeza no dataset na ordem correta.
        """
        logger.info("Iniciando processo de limpeza do dataset...")
        
        # Remover colunas irrelevantes
        df = self.remove_irrelevant_columns(df)
        
        # Tratar valores ausentes
        df = self.handle_missing_values(df)
        
        # Remover duplicatas
        df = self.remove_duplicates(df)
        
        logger.info("Processo de limpeza concluído com sucesso!")
        return df