import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.feature_extraction import FeatureHasher
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureSelection:
    _label_encoders = {}
    _onehot_encoder = {}
    _target_encoder = {}
    _scaler = StandardScaler()
    
    @staticmethod
    def remove_id_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas de ID do DataFrame."""
        id_columns = ['id']
        initial_cols = df.columns.tolist()
        df = df.drop(columns=id_columns, errors='ignore')
        removed_cols = set(initial_cols) - set(df.columns)
        
        if removed_cols:
            logger.info(f"Colunas de ID removidas: {removed_cols}")
        return df

    @staticmethod
    def apply_onehot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Aplica One-Hot Encoding para variáveis categóricas com baixa cardinalidade."""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                if col not in FeatureSelection._onehot_encoder:
                    FeatureSelection._onehot_encoder[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')

                    FeatureSelection._onehot_encoder[col].fit(df[[col]])
                
                encoded_array = FeatureSelection._onehot_encoder[col].transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in FeatureSelection._onehot_encoder[col].categories_[0]]
                
                for i, name in enumerate(feature_names):
                    df_encoded[name] = encoded_array[:, i]
                
                df_encoded = df_encoded.drop(columns=[col])
                logger.info(f"One-Hot Encoding aplicado na coluna {col}, gerando {len(feature_names)} novas features")
        
        return df_encoded
    
    @staticmethod
    def apply_label_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Aplica Label Encoding para variáveis categóricas ordinais."""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                if col not in FeatureSelection._label_encoders:
                    FeatureSelection._label_encoders[col] = LabelEncoder()
                    FeatureSelection._label_encoders[col].fit(df[col].astype(str))
                
                df_encoded[col] = FeatureSelection._label_encoders[col].transform(df[col].astype(str))
                logger.info(f"Label Encoding aplicado na coluna: {col}")
        
        return df_encoded

    @staticmethod
    def apply_target_encoding(df: pd.DataFrame, columns: list, target_col: str = 'severidade') -> pd.DataFrame:
        """Aplica Target Encoding para variáveis com alta cardinalidade."""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                if col not in FeatureSelection._target_encoder:
                    FeatureSelection._target_encoder[col] = TargetEncoder()
                    if target_col in df.columns:
                        FeatureSelection._target_encoder[col].fit(df[col], df[target_col])
                    else:
                        logger.warning(f"Coluna target '{target_col}' não encontrada. Usando target encoding sem supervisão.")
                        FeatureSelection._target_encoder[col].fit(df[col])
                
                df_encoded[col] = FeatureSelection._target_encoder[col].transform(df[col])
                logger.info(f"Target Encoding aplicado na coluna: {col}")
        
        return df_encoded
    
    @staticmethod
    def standardize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza features numéricas"""
        df_scaled = df.copy()
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_columns = [col for col in numeric_columns]
        
        if numeric_columns:
            df_scaled[numeric_columns] = FeatureSelection._scaler.fit_transform(df[numeric_columns])
            logger.info(f"Padronização aplicada nas colunas: {numeric_columns}")
            logger.info(f"Média das colunas após padronização: {df_scaled[numeric_columns].mean().round(2).to_dict()}")
            logger.info(f"Desvio padrão após padronização: {df_scaled[numeric_columns].std().round(2).to_dict()}")
        
        return df_scaled
    
    @staticmethod
    def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas as transformações de features no dataset."""
        logger.info("Iniciando transformação de features...")
        initial_shape = df.shape
        
        onehot_columns = [
            'sentido_via',
            'tipo_pista',
            'fase_dia',
            'classificacao_acidente',
            'uso_solo'
        ]
        
        label_encoding_columns = [
            'dia_semana'
        ]
        
        target_encoding_columns = [
            'municipio',
            'causa_acidente',
            'tracado_via'
        ]
        
        df = FeatureSelection.remove_id_columns(df)
        df = FeatureSelection.apply_onehot_encoding(df, onehot_columns)
        df = FeatureSelection.apply_label_encoding(df, label_encoding_columns)
        df = FeatureSelection.apply_target_encoding(df, target_encoding_columns)
        df = FeatureSelection.standardize_numeric_features(df)
        
        logger.info(f"Shape inicial do dataset: {initial_shape}")
        logger.info(f"Shape final do dataset: {df.shape}")
        logger.info(f"Colunas finais: {df.columns.tolist()}")
        
        return df