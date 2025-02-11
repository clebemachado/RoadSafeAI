from .data_balancing_strategy import DataBalancingStrategy
from imblearn.under_sampling import RandomUnderSampler

class RandomUndersamplingBalancing(DataBalancingStrategy):
    def apply(self, X, y, random_state = 42):
        self.logger.info("Aplicando Random Undersampling...")
        self.log_class_distribution(y, "antes do undersampling")
        
        undersample = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = undersample.fit_resample(X, y)
        
        self.log_class_distribution(y_balanced, "após undersampling")
        return X_balanced, y_balanced