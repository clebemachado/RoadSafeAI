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
    def select_features(df, threshold=0.9):
        """ Remove colunas altamente correlacionadas """
        
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        df = df.drop(columns=to_drop)
        
        return df