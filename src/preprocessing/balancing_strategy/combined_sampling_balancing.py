from .data_balancing_strategy import DataBalancingStrategy 
from .smote_balancing import SmoteBalancing
from .random_oversampling_balancing import RandomOversamplingBalancing

class CombinedSamplingBalancing(DataBalancingStrategy):
    def apply(self, X, y, random_state = 42):
        self.logger.info("Aplicando estratégia combinada de sampling...")
        # `DataBalance` is a class that provides methods for balancing data in a dataset of accidents using different strategies such as oversampling and undersampling. It implements the following functionalities:
        self.log_class_distribution(y, "original")
        
        # Primeiro aplica undersampling nas classes majoritárias
        random_undersampling: RandomOversamplingBalancing = RandomOversamplingBalancing()
        X_under, y_under = random_undersampling.apply(X, y, random_state)
        
        # Depois aplica SMOTE para balancear todas as classes
        smote: SmoteBalancing = SmoteBalancing()
        X_balanced, y_balanced = smote.apply(X_under, y_under, random_state)
        
        return X_balanced, y_balanced