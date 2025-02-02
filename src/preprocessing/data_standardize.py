import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataStandardize:
    
    @staticmethod
    def adjust_numeric_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
        """Ajusta o tipo de dado de uma coluna para o tipo desejado, tratando valores nulos."""
        initial_type = df[column].dtype
        try:
            if df[column].isnull().any():
                df[column] = df[column].fillna(0)

            # Se for int, usar 'Int64' para suportar NaN
            if target_type == 'int64':
                df[column] = df[column].astype('Int64')
            else:
                df[column] = df[column].astype(target_type)

            logger.info(f"A coluna '{column}' foi convertida de {initial_type} para {df[column].dtype}.")
        except Exception as e:
            logger.error(f"Erro ao tentar converter a coluna '{column}' para {target_type}: {e}")
        return df

    @staticmethod
    def convert_to_datetime(df: pd.DataFrame, date_column: str, time_column: str) -> pd.DataFrame:
        """Converte colunas de data e horário para datetime/time, tratando NaNs."""
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            if df[time_column].notna().any():
                df[time_column] = pd.to_datetime(df[time_column], format='%H:%M:%S', errors='coerce').dt.time

            logger.info(f"As colunas '{date_column}' e '{time_column}' foram convertidas corretamente.")
        except Exception as e:
            logger.error(f"Erro ao converter '{date_column}' e '{time_column}': {e}")
        return df


    @staticmethod
    def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas as funções de padronização no dataset."""
        # Ajustando tipo de dados para a coluna 'br' como exemplo
        df = DataStandardize.adjust_numeric_type(df, 'br', 'int64')
    
        # Convertendo as colunas de data e horário
        df = DataStandardize.convert_to_datetime(df, 'data_inversa', 'horario')
        
        return df