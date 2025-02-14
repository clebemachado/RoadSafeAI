from sklearn.feature_selection import SelectKBest, chi2

class FeatureSelection:
    """Classe para seleção de características."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def select_k_best(self, k):
        """Seleciona as k melhores características usando o teste qui-quadrado."""
        selector = SelectKBest(score_func=chi2, k=k)
        self.X = selector.fit_transform(self.X, self.y)
        return self.X
