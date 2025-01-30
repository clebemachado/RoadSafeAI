import pandas as pd
import numpy as np
from typing import Dict, List

class WeatherAnalysis:
    """Análises do impacto das condições meteorológicas nos acidentes."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def get_weather_stats(self) -> pd.DataFrame:
        """Estatísticas de acidentes por condição meteorológica."""
        return self.df.groupby('condicao_metereologica').agg({
            'id': 'count',
            'mortos': ['sum', 'mean'],
            'feridos': ['sum', 'mean']
        }).round(2).sort_values(('id', 'count'), ascending=False)
    
    def get_severity_by_weather(self) -> pd.DataFrame:
        """Severidade média dos acidentes por condição meteorológica."""
        self.df['indice_severidade'] = (
            self.df['mortos'] * 13 + self.df['feridos_graves'] * 5 +
            self.df['feridos_leves'] * 1
        ) / self.df['veiculos']
        
        return self.df.groupby('condicao_metereologica').agg({
            'indice_severidade': 'mean',
            'mortos': ['sum', 'mean'],
            'feridos_graves': ['sum', 'mean'],
            'feridos_leves': ['sum', 'mean']
        }).round(2).sort_values('indice_severidade', ascending=False)
    
    def get_weather_trend(self) -> pd.DataFrame:
        """Tendências de acidentes ao longo dos anos para diferentes condições meteorológicas."""
        self.df['ano'] = pd.to_datetime(self.df['data_inversa']).dt.year
        return self.df.groupby(['ano', 'condicao_metereologica']).size().unstack().fillna(0)