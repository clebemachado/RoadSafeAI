import numpy as np

class FeatureTransformation:
    """Classe para transformação de variáveis."""

    def __init__(self, df):
        self.df = df

    def log_transform(self, column):
        """Aplica transformação logarítmica a uma coluna numérica."""
        self.df[column] = np.log1p(self.df[column])

    def sqrt_transform(self, column):
        """Aplica transformação raiz quadrada a uma coluna numérica."""
        self.df[column] = np.sqrt(self.df[column])

    def get_transformed_data(self):
        """Retorna o DataFrame com as transformações aplicadas."""
        return self.df
