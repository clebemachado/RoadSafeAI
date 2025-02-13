import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from config.inject_logger import inject_logger


@inject_logger
class DataSplit:
    """
    Classe responsável pela separação dos dados.
    Target: gravidade_acidente
    """
    
    # Colunas que não devem ser usadas como features
    COLUMNS_TO_DROP = [
        'Unnamed: 0',  # Índice
        'data',         # Data já está representada em outras features
        'causa_acidente' # Usar causa acidente agrupado
    ]
    
    def remove_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas que não serão utilizadas no modelo.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame sem as colunas não utilizadas
        """
        df = df.copy()
        columns_to_drop = [col for col in DataSplit.COLUMNS_TO_DROP if col in df.columns]
        
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Colunas removidas: {columns_to_drop}")
        
        return df

    def check_target_distribution(self, y: pd.Series, set_name: str = "") -> None:
        """
        Verifica e loga a distribuição das classes no target.
        
        Args:
            y: Série com os valores do target
            set_name: Nome do conjunto de dados (para logging)
        """
        dist = y.value_counts(normalize=True) * 100
        self.logger.info(f"\nDistribuição das classes{' - ' + set_name if set_name else ''}:")
        for classe, prop in dist.items():
            self.logger.info(f"- Classe {classe}: {prop:.2f}%")

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Realiza a separação dos dados em conjuntos de treino, validação e teste.
        
        Args:
            df: DataFrame preparado
            test_size: Proporção do conjunto de teste
            valid_size: Proporção do conjunto de validação
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple contendo:
            - X_train, X_valid, X_test: Features para treino, validação e teste
            - y_train, y_valid, y_test: Target para treino, validação e teste
        """
        # Separar features e target
        X = df.drop('gravidade_acidente', axis=1)
        y = df['gravidade_acidente']
        
        # Verificar distribuição inicial das classes
        self.logger.info("Distribuição inicial das classes:")
        self.check_target_distribution(y)
        
        # Primeiro split: separa o conjunto de teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Segundo split: separa treino e validação
        valid_size_adjusted = valid_size / (1 - test_size)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=valid_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Log das dimensões dos conjuntos
        self.logger.info("\nDimensões dos conjuntos:")
        self.logger.info(f"- Treino: {X_train.shape}")
        self.logger.info(f"- Validação: {X_valid.shape}")
        self.logger.info(f"- Teste: {X_test.shape}")
        
        # Verificar distribuição das classes em cada conjunto
        self.check_target_distribution(y_train, "Treino")
        self.check_target_distribution(y_valid, "Validação")
        self.check_target_distribution(y_test, "Teste")
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Executa todo o pipeline de preparação dos dados.
        
        Args:
            df: DataFrame original
            test_size: Proporção do conjunto de teste
            valid_size: Proporção do conjunto de validação
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple contendo:
            - X_train, X_valid, X_test: Features para treino, validação e teste
            - y_train, y_valid, y_test: Target para treino, validação e teste
        """
        self.logger.info("Iniciando preparação dos dados...")
        
        # Remover colunas não utilizadas
        df = self.remove_unused_columns(df)
        
        # Split dos dados
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data(
            df,
            test_size=test_size,
            valid_size=valid_size,
            random_state=random_state
        )
        
        self.logger.info("Preparação dos dados concluída!")
        return X_train, X_valid, X_test, y_train, y_valid, y_test

# Exemplo de uso:
"""
import pandas as pd
from decision_tree_split import DataSplit

# Carregar dados
df = pd.read_csv('df_encoded_ma.csv')

# Preparar dados
X_train, X_valid, X_test, y_train, y_valid, y_test = DataSplit.prepare_data(df)

# Os dados estão prontos para serem utilizados no treinamento da árvore de decisão
"""