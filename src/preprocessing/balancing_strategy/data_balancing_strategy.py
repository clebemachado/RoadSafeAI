import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from collections import Counter
from collections import Counter
from typing import Tuple, Union
from config.inject_logger import inject_logger


@inject_logger
class DataBalancingStrategy(ABC):
    
    @abstractmethod
    def apply(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        pass
    
    def log_class_distribution(self, y: Union[pd.Series, np.ndarray], stage: str = "") -> None:
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
        
        self.logger.info(f"\nDistribuição das classes{' - ' + stage if stage else ''}:")
        for classe, count in distribution.items():
            percentage = (count/total) * 100
            self.logger.info(f"- Classe {classe}: {count} amostras ({percentage:.2f}%)")