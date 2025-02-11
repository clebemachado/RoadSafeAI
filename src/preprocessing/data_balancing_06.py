import pandas as pd
import numpy as np
from typing import Tuple
from config.inject_logger import inject_logger
from preprocessing.balancing_strategy import DataBalancingStrategy, SmoteBalancing, RandomOversamplingBalancing, RandomUndersamplingBalancing, CombinedSamplingBalancing


VALID_STRATEGIES = ['smote', 'random_over', 'random_under', 'combined']

@inject_logger
class DataBalance:
    """
    Classe responsável pelo aplicação do balanceamento dos dados do dataset de acidentes.
    Implementa diferentes estratégias de balanceamento (oversampling e undersampling).
    """
    def balance_data(
        self,
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
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Estratégia {strategy} inválida. Use uma das seguintes: {VALID_STRATEGIES}"
            )
        
        balancing_strategy: DataBalancingStrategy = None
        
        if strategy == 'smote':
            balancing_strategy = SmoteBalancing()
        elif strategy == 'random_over':
            balancing_strategy = RandomOversamplingBalancing()
        elif strategy == 'random_under':
            balancing_strategy = RandomUndersamplingBalancing
        elif strategy == "combined":  # combined
            balancing_strategy = CombinedSamplingBalancing()
        else:
            raise ValueError(
                f"Estratégia {strategy} inválida. Use uma das seguintes: {VALID_STRATEGIES}"
            )
        
        return balancing_strategy.apply(X=X, y=y, random_state=random_state)