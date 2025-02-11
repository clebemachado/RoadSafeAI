import logging
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from config.inject_logger import inject_logger


@inject_logger
class DataEncoding:
    """
    Classe responsável pelo encoding de variáveis categóricas.
    Estratégias:
    - OneHot: Para colunas com poucas categorias
    - Label: Para colunas com ordem implícita
    - Target: Para colunas com muitas categorias
    """
    
    # Colunas que possuem ordem implícita para Label Encoding
    ORDINAL_COLUMNS = [
        'periodo_dia',         # madrugada -> manhã -> tarde -> noite
        'dia_semana'          # segunda -> terça -> ... -> domingo
    ]
    
    # Mapeamento de ordem para colunas ordinais
    ORDINAL_MAPPING = {
        'periodo_dia': {
            'madrugada': 0,
            'manha': 1,
            'tarde': 2,
            'noite': 3
        },
        'dia_semana': {
            'segunda-feira': 0,
            'terca-feira': 1,
            'quarta-feira': 2,
            'quinta-feira': 3,
            'sexta-feira': 4,
            'sabado': 5,
            'domingo': 6
        }
    }
    
    # Colunas que devem ser removidas antes do encoding
    _COLUMNS_TO_REMOVE = [
        'feridos_graves',
        'mortos',
        'feridos_leves',
        'ilesos'
    ]
    
    def __init__(self):
        """Inicializa os encoders e lista de colunas a serem removidas."""
        self.label_encoders = {}
        self.onehot_encoder = None
        self.target_encoders = {}
        self.onehot_features = None
        self.target_features = None
        self.columns_to_remove = self._COLUMNS_TO_REMOVE.copy()
    
    def set_columns_to_remove(self, columns: List[str], append: bool = False):
        """
        Define as colunas que devem ser removidas antes do encoding.
        
        Args:
            columns: Lista de nomes das colunas a serem removidas
            append: Se True, adiciona as colunas à lista padrão. Se False, substitui a lista padrão
        """
        if append:
            self.columns_to_remove.extend(columns)
        else:
            self.columns_to_remove = columns
        self.logger.info(f"Definidas {len(self.columns_to_remove)} colunas para remoção: {self.columns_to_remove}")
    
    def _remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove as colunas especificadas do DataFrame.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame sem as colunas removidas
        """
        df_cleaned = df.copy()
        columns_present = [col for col in self.columns_to_remove if col in df_cleaned.columns]
        
        if columns_present:
            df_cleaned = df_cleaned.drop(columns=columns_present)
            self.logger.info(f"Removidas as colunas: {columns_present}")
        
        return df_cleaned

    def _identify_categorical_columns(self, df: pd.DataFrame, target_column: str,
                                   max_categories_onehot: int = 5) -> Tuple[List[str], List[str]]:
        """
        Identifica colunas categóricas e decide a estratégia de encoding.
        
        Args:
            df: DataFrame
            target_column: Nome da coluna target que não deve ser encodada
            max_categories_onehot: Número máximo de categorias para usar OneHot
            
        Returns:
            Tuple com listas de colunas para OneHot e Target Encoding
        """
        onehot_columns = []
        target_columns = []
        
        for col in df.select_dtypes(include=['object', 'category']).columns:
            # Pula a coluna target e colunas ordinais
            if col == target_column or col in self.ORDINAL_COLUMNS:
                continue
                
            n_unique = df[col].nunique()
            self.logger.info(f"Coluna {col}: {n_unique} valores únicos")
            
            if n_unique <= max_categories_onehot:
                onehot_columns.append(col)
            else:
                target_columns.append(col)
        
        return onehot_columns, target_columns

    def fit(self, df: pd.DataFrame, target_column: str, max_categories_onehot: int = 5):
        """
        Ajusta os encoders aos dados.
        
        Args:
            df: DataFrame
            target_column: Nome da coluna target que não deve ser encodada
            max_categories_onehot: Número máximo de categorias para usar OneHot
        """
        self.logger.info("Iniciando fit dos encoders...")
        
        # Remover colunas especificadas
        df_cleaned = self._remove_columns(df)
        
        # Identificar colunas categóricas e estratégias
        self.onehot_features, self.target_features = self._identify_categorical_columns(
            df_cleaned, target_column, max_categories_onehot
        )
        
        # Ajustar Label Encoders para colunas ordinais
        for col in self.ORDINAL_COLUMNS:
            if col in df_cleaned.columns:
                self.label_encoders[col] = LabelEncoder()
                # Garantir que o encoder respeite a ordem definida
                ordered_categories = list(self.ORDINAL_MAPPING[col].keys())
                self.label_encoders[col].fit(ordered_categories)
                self.logger.info(f"Label Encoder ajustado para {col}")
        
        # Ajustar OneHot Encoder
        if self.onehot_features:
            self.onehot_encoder = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False), self.onehot_features)
                ],
                remainder='passthrough'
            )
            self.onehot_encoder.fit(df_cleaned[self.onehot_features])
            self.logger.info(f"OneHot Encoder ajustado para {self.onehot_features}")
        
        # Ajustar Target Encoders
        for col in self.target_features:
            self.target_encoders[col] = TargetEncoder()
            self.target_encoders[col].fit(df_cleaned[[col]], df_cleaned[target_column])
            self.logger.info(f"Target Encoder ajustado para {col}")

    def transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Aplica as transformações no DataFrame.
        
        Args:
            df: DataFrame original
            target_column: Nome da coluna target que não deve ser encodada
            
        Returns:
            DataFrame com encodings aplicados
        """
        # Remover colunas especificadas
        df_transformed = self._remove_columns(df)
        
        # Aplicar Label Encoding para colunas ordinais
        for col, encoder in self.label_encoders.items():
            if col in df_transformed.columns:
                df_transformed[col] = encoder.transform(df_transformed[col].astype(str))
                self.logger.info(f"Label Encoding aplicado em {col}")
        
        # Aplicar OneHot Encoding
        if self.onehot_features:
            onehot_array = self.onehot_encoder.transform(df_transformed[self.onehot_features])
            
            # Criar nomes para as novas colunas
            feature_names = []
            for i, feature in enumerate(self.onehot_features):
                categories = self.onehot_encoder.named_transformers_['onehot'].categories_[i][1:]
                feature_names.extend([f"{feature}_{cat}" for cat in categories])
            
            # Substituir colunas originais pelas codificadas
            df_transformed = df_transformed.drop(columns=self.onehot_features)
            onehot_df = pd.DataFrame(onehot_array, columns=feature_names, index=df_transformed.index)
            df_transformed = pd.concat([df_transformed, onehot_df], axis=1)
            self.logger.info(f"OneHot Encoding aplicado em {self.onehot_features}")
        
        # Aplicar Target Encoding
        for col, encoder in self.target_encoders.items():
            if col in df_transformed.columns:
                transformed_values = encoder.transform(df_transformed[[col]])
                if transformed_values.ndim > 1:
                    transformed_values = transformed_values.mean(axis=1)
                df_transformed[col] = transformed_values
                self.logger.info(f"Target Encoding aplicado em {col}")
        
        return df_transformed

    def fit_transform(self, df: pd.DataFrame, target_column: str, 
                     max_categories_onehot: int = 5) -> pd.DataFrame:
        """
        Ajusta os encoders e aplica as transformações.
        
        Args:
            df: DataFrame original
            target_column: Nome da coluna target que não deve ser encodada
            max_categories_onehot: Número máximo de categorias para usar OneHot
            
        Returns:
            DataFrame com encodings aplicados
        """
        self.fit(df, target_column, max_categories_onehot)
        return self.transform(df, target_column)

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Retorna os nomes das features após as transformações.
        
        Returns:
            Dicionário com os nomes das features por tipo de transformação
        """
        feature_names = {
            'ordinal_encoded': list(self.label_encoders.keys()),
            'onehot_encoded': self.onehot_features if self.onehot_features else [],
            'target_encoded': list(self.target_encoders.keys()),
            'removed_columns': self.columns_to_remove
        }
        return feature_names