import pandas as pd

class FeatureEngineering:
    """Classe para engenharia de atributos."""

    def __init__(self, df):
        self.df = df

    def extract_date_features(self, date_column):
        """Cria novas colunas a partir de uma coluna de data."""
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df["year"] = self.df[date_column].dt.year
        self.df["month"] = self.df[date_column].dt.month
        self.df["day"] = self.df[date_column].dt.day
        self.df["weekday"] = self.df[date_column].dt.weekday

    def get_data(self):
        """Retorna o DataFrame com novas features."""
        return self.df
    def create_features(df):
        """ Gera novas features derivadas """
        
        df['hora_do_dia'] = pd.to_datetime(df['data_inversa']).dt.hour
        df['dia_da_semana'] = pd.to_datetime(df['data_inversa']).dt.day_name()
        
        return df
