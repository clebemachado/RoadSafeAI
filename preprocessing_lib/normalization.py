from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Normalization:
    """Classe para normalização e padronização de dados numéricos."""

    def __init__(self, df):
        self.df = df

    def min_max_scale(self, columns):
        """Aplica Min-Max Scaling para normalizar valores entre 0 e 1."""
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])

    def standardize(self, columns):
        """Aplica padronização (z-score) às colunas especificadas."""
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])

    def get_scaled_data(self):
        """Retorna o DataFrame normalizado ou padronizado."""
        return self.df
