import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataStandardize:
    """
    Classe responsável pela padronização dos dados do dataset de acidentes.
    """
    
    # Colunas numéricas que precisam ser verificadas/ajustadas
    NUMERIC_COLUMNS = [
        'km', 'pessoas', 'mortos', 'feridos_leves',
        'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos'
    ]
    
    # Mapeamento para uso_solo
    USO_SOLO_MAPPING = {
        'Rural': 'Não',
        'Urbano': 'Sim',
        'Não': 'Não',
        'Sim': 'Sim'
    }
    
    # Mapeamento para dias da semana
    DIAS_SEMANA_MAPPING = {
        'Segunda': 'segunda-feira',
        'Terça': 'terca-feira',
        'Quarta': 'quarta-feira',
        'Quinta': 'quinta-feira',
        'Sexta': 'sexta-feira',
        'Sábado': 'sabado',
        'Domingo': 'domingo',
        'domingo': 'domingo',
        'segunda-feira': 'segunda-feira',
        'terça-feira': 'terca-feira',
        'quarta-feira': 'quarta-feira',
        'quinta-feira': 'quinta-feira',
        'sexta-feira': 'sexta-feira',
        'sábado': 'sabado'
    }
    
    # Mapeamento para padronização de texto
    TEXT_STANDARDIZATION = {
        'tipo_acidente': {
            'Colisão Transversal': 'colisao transversal',
            'Colisão com objeto fixo': 'colisao com objeto estatico',
            'Colisão lateral mesmo sentido': 'colisao lateral',
            'Colisão lateral sentido oposto': 'colisao lateral',
            'Atropelamento de Pedestre': 'atropelamento de pessoa',
            'Saída de Pista': 'saida de leito carrocavel'
        },
        'classificacao_acidente': {
            'Com Vítimas Fatais': 'com vitimas fatais',
            'Com Vítimas Feridas': 'com vitimas feridas',
            'Sem Vítimas': 'sem vitimas',
            'Ignorado': 'ignorado'
        }
    }

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
    def padronizar_valores_temporais(df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza as colunas temporais do dataset.
        - Converte data_inversa para datetime
        - Cria coluna periodo_dia baseada no horário
        
        Args:
            df: DataFrame com as colunas temporais
            
        Returns:
            DataFrame com valores temporais padronizados
        """
        df = df.copy()
        
        # Converter data_inversa para datetime
        if 'data_inversa' in df.columns:
            df['data'] = pd.to_datetime(df['data_inversa'])
            df['ano'] = df['data'].dt.year
            df.drop('data_inversa', axis=1, inplace=True)
            logger.info("Coluna data_inversa convertida para datetime e renomeada para 'data'")
        
        return df

    @staticmethod
    def padronizar_uso_solo(df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza a coluna uso_solo conforme dicionário da PRF.
        Rural -> Não
        Urbano -> Sim
        
        Args:
            df: DataFrame com a coluna uso_solo
            
        Returns:
            DataFrame com uso_solo padronizado
        """
        df = df.copy()
        
        if 'uso_solo' in df.columns:
            # Registrar valores únicos antes da transformação
            valores_originais = df['uso_solo'].unique()
            logger.info(f"Valores únicos originais em uso_solo: {valores_originais}")
            
            # Aplicar mapeamento
            df['uso_solo'] = df['uso_solo'].replace(DataStandardize.USO_SOLO_MAPPING)
            
            # Registrar valores únicos após a transformação
            valores_finais = df['uso_solo'].unique()
            logger.info(f"Valores únicos após padronização em uso_solo: {valores_finais}")
            
            # Verificar se existem valores não mapeados
            valores_nao_mapeados = set(valores_originais) - set(DataStandardize.USO_SOLO_MAPPING.keys())
            if valores_nao_mapeados:
                logger.warning(f"Valores não mapeados em uso_solo: {valores_nao_mapeados}")
        
        return df

    @staticmethod
    def padronizar_dia_semana(df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza os valores da coluna dia_semana para um formato único.
        
        Args:
            df: DataFrame com a coluna dia_semana
            
        Returns:
            DataFrame com dia_semana padronizado
        """
        df = df.copy()
        
        if 'dia_semana' in df.columns:
            # Registrar valores únicos antes da transformação
            valores_originais = df['dia_semana'].unique()
            logger.info(f"Valores únicos originais em dia_semana: {valores_originais}")
            
            # Aplicar mapeamento
            df['dia_semana'] = df['dia_semana'].replace(DataStandardize.DIAS_SEMANA_MAPPING)
            
            # Registrar valores únicos após a transformação
            valores_finais = df['dia_semana'].unique()
            logger.info(f"Valores únicos após padronização em dia_semana: {valores_finais}")
            
            # Verificar se existem valores não mapeados
            valores_nao_mapeados = set(valores_originais) - set(DataStandardize.DIAS_SEMANA_MAPPING.keys())
            if valores_nao_mapeados:
                logger.warning(f"Valores não mapeados em dia_semana: {valores_nao_mapeados}")
        
        return df

    @staticmethod
    def standardize_text(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Padroniza o texto nas colunas especificadas usando o mapeamento definido.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns and col in DataStandardize.TEXT_STANDARDIZATION:
                # Converter para lowercase
                df[col] = df[col].str.lower()
                # Aplicar mapeamento
                df[col] = df[col].replace(
                    {k.lower(): v for k, v in DataStandardize.TEXT_STANDARDIZATION[col].items()}
                )
                logger.info(f"Coluna {col} padronizada. Valores únicos: {df[col].nunique()}")
        return df

    @classmethod
    def padronizar_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as padronizações no dataset.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame padronizado
        """
        logger.info("Iniciando padronização do dataset...")
        
        # Padronizar valores numéricos
        df = self.adjust_numeric_type(df)
        logger.info("Valores numéricos padronizados")
        
        # Padronizar valores temporais
        df = self.padronizar_valores_temporais(df)
        logger.info("Valores temporais padronizados")
        
        # Padronizar uso_solo
        df = self.padronizar_uso_solo(df)
        logger.info("Coluna uso_solo padronizada")
        
        # Padronizar texto nas colunas especificadas
        df = self.standardize_text(df, ['tipo_acidente', 'classificacao_acidente'])
        logger.info("Coluna tipo_acidente e classificacao_acidente padronizada")
        
        # Padronizar dia_semana
        df = self.padronizar_dia_semana(df)
        logger.info("Coluna dia_semana padronizada")
        
        return df
