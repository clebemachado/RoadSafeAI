import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import logging
from typing import Tuple, Dict, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataBalance:
    """
    Classe responsável pelo balanceamento dos dados do dataset de acidentes.
    Implementa diferentes estratégias de balanceamento (oversampling e undersampling).
    """
    
    VALID_STRATEGIES = ['smote', 'random_over', 'random_under', 'combined']
    
    @staticmethod
    def log_class_distribution(y: Union[pd.Series, np.ndarray], stage: str = "") -> None:
        """
        Loga a distribuição das classes.
        
        Args:
            y: Array ou Series com as classes
            stage: Identificador do estágio (para logging)
        """
        if isinstance(y, pd.Series):
            distribution = y.value_counts()
        else:
            distribution = pd.Series(Counter(y))
            
        total = len(y)
        logger.info(f"\nDistribuição das classes{' - ' + stage if stage else ''}:")
        for classe, count in distribution.items():
            percentage = (count/total) * 100
            logger.info(f"- Classe {classe}: {count} amostras ({percentage:.2f}%)")

    @staticmethod
    def apply_smote(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica SMOTE para balancear as classes.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        logger.info("Aplicando SMOTE...")
        DataBalance.log_class_distribution(y, "antes do SMOTE")
        
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        DataBalance.log_class_distribution(y_balanced, "após SMOTE")
        return X_balanced, y_balanced

    @staticmethod
    def apply_random_oversampling(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica Random Oversampling para balancear as classes.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        logger.info("Aplicando Random Oversampling...")
        DataBalance.log_class_distribution(y, "antes do oversampling")
        
        oversample = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = oversample.fit_resample(X, y)
        
        DataBalance.log_class_distribution(y_balanced, "após oversampling")
        return X_balanced, y_balanced

    @staticmethod
    def apply_random_undersampling(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica Random Undersampling para balancear as classes.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        logger.info("Aplicando Random Undersampling...")
        DataBalance.log_class_distribution(y, "antes do undersampling")
        
        undersample = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = undersample.fit_resample(X, y)
        
        DataBalance.log_class_distribution(y_balanced, "após undersampling")
        return X_balanced, y_balanced

    @staticmethod
    def apply_combined_sampling(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica uma combinação de under e oversampling para balancear as classes.
        Primeiro reduz as classes majoritárias e depois aumenta as minoritárias.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        logger.info("Aplicando estratégia combinada de sampling...")
        DataBalance.log_class_distribution(y, "original")
        
        # Primeiro aplica undersampling nas classes majoritárias
        X_under, y_under = DataBalance.apply_random_undersampling(X, y, random_state)
        
        # Depois aplica SMOTE para balancear todas as classes
        X_balanced, y_balanced = DataBalance.apply_smote(X_under, y_under, random_state)
        
        return X_balanced, y_balanced

    @classmethod
    def balance_data(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = 'smote',
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Método principal para balanceamento dos dados.
        
        Args:
            X: Features
            y: Target
            strategy: Estratégia de balanceamento ('smote', 'random_over', 'random_under', 'combined')
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        if strategy not in cls.VALID_STRATEGIES:
            raise ValueError(
                f"Estratégia {strategy} inválida. Use uma das seguintes: {cls.VALID_STRATEGIES}"
            )
        
        if strategy == 'smote':
            return cls.apply_smote(X, y, random_state)
        elif strategy == 'random_over':
            return cls.apply_random_oversampling(X, y, random_state)
        elif strategy == 'random_under':
            return cls.apply_random_undersampling(X, y, random_state)
        else:  # combined
            return cls.apply_combined_sampling(X, y, random_state)

# Exemplo de uso:
"""
from data_balance import DataBalance

# Assumindo que você já tem X_train e y_train
# Aplicar balanceamento usando SMOTE
X_train_balanced, y_train_balanced = DataBalance.balance_data(
    X_train,
    y_train,
    strategy='smote'
)

# Ou usando outra estratégia
X_train_balanced, y_train_balanced = DataBalance.balance_data(
    X_train,
    y_train,
    strategy='combined'
)
"""