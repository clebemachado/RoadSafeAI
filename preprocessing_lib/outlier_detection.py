import numpy as np

class OutlierDetection:
    """Classe para detecção e remoção de outliers."""

    def __init__(self, df):
        self.df = df

    def remove_outliers_iqr(self, column):
        """Remove outliers com base no método do IQR (Interquartile Range)."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]

    def get_data(self):
        """Retorna o DataFrame sem outliers."""
        return self.df
    
    def detect_outliers(df, column, method="iqr"):
        """ Detecta outliers com diferentes métodos """
        
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
        
        elif method == "zscore":
            mean = np.mean(df[column])
            std = np.std(df[column])
            outliers = df[np.abs((df[column] - mean) / std) > 3]
        
        return outliers
