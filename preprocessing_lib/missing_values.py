class MissingValues:
    """Classe para tratamento de valores ausentes."""

    def __init__(self, df):
        self.df = df

    def fill_missing_with_mean(self, column):
        """Preenche valores ausentes com a média da coluna."""
        self.df[column].fillna(self.df[column].mean(), inplace=True)

    def fill_missing_with_median(self, column):
        """Preenche valores ausentes com a mediana da coluna."""
        self.df[column].fillna(self.df[column].median(), inplace=True)

    def fill_missing_with_mode(self, column):
        """Preenche valores ausentes com a moda (valor mais frequente) da coluna."""
        self.df[column].fillna(self.df[column].mode()[0], inplace=True)

    def get_data(self):
        """Retorna o DataFrame com valores ausentes tratados."""
        return self.df
