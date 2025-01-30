import pandas as pd

class FeatureEngineering:
    """
    Classe responsável por criar novas features a partir dos dados brutos.
    """
    
    @staticmethod
    def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features temporais a partir da coluna de data e horário."""
        df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
        df['ano'] = df['data_inversa'].dt.year
        df['mes'] = df['data_inversa'].dt.month
        df['dia'] = df['data_inversa'].dt.day
        df['dia_semana_num'] = df['data_inversa'].dt.weekday
        df['hora'] = pd.to_datetime(df['horario'], errors='coerce').dt.hour
        return df
    
    @staticmethod
    def categorize_severity(df: pd.DataFrame) -> pd.DataFrame:
        """Cria uma categoria de severidade do acidente baseada no número de mortos e feridos."""
        df['severidade'] = 'leve'
        df.loc[df['feridos_graves'] > 0, 'severidade'] = 'grave'
        df.loc[df['mortos'] > 0, 'severidade'] = 'fatal'
        return df
    
    @staticmethod
    def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis binárias para condições meteorológicas e tipo de pista."""
        df['chuva'] = df['condicao_metereologica'].str.contains('Chuva', na=False).astype(int)
        df['neblina_nevoeiro'] = df['condicao_metereologica'].str.contains('Nevoeiro|Neblina', na=False).astype(int)
        df['pista_dupla'] = df['tipo_pista'].str.contains('Dupla', na=False).astype(int)
        return df
    
    @staticmethod
    def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cria agregações como total de vítimas por acidente."""
        df['total_vitimas'] = df['mortos'] + df['feridos_leves'] + df['feridos_graves']
        return df
    
    @staticmethod
    def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Executa todo o pipeline de engenharia de features."""
        df = FeatureEngineering.extract_datetime_features(df)
        df = FeatureEngineering.categorize_severity(df)
        df = FeatureEngineering.create_binary_features(df)
        df = FeatureEngineering.aggregate_features(df)
        return df