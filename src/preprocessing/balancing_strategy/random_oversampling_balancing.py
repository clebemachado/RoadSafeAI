from imblearn.over_sampling import RandomOverSampler
from .data_balancing_strategy import DataBalancingStrategy

class RandomOversamplingBalancing(DataBalancingStrategy):
    def apply(self, X, y, random_state = 42):
        """
        Aplica Random Oversampling para balancear as classes.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        self.logger.info("Aplicando Random Oversampling...")
        self.log_class_distribution(y, "antes do oversampling")
        
        oversample = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = oversample.fit_resample(X, y)
        
        self.log_class_distribution(y_balanced, "após oversampling")
        return X_balanced, y_balanced