from imblearn.over_sampling import SMOTE

from .data_balancing_strategy import DataBalancingStrategy

class SmoteBalancing(DataBalancingStrategy):
    def apply(self, X, y, random_state = 42):
        """
        Aplica SMOTE para balancear as classes.
        
        Args:
            X: Features
            y: Target
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com X e y balanceados
        """
        self.logger.info("Aplicando SMOTE...")
        self.log_class_distribution(y, "antes do SMOTE")
        
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        self.log_class_distribution(y_balanced, "após SMOTE")
        return X_balanced, y_balanced