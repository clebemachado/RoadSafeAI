import pandas as pd
import numpy as np
from typing import Dict, List

class TrendAnalysis:
    """Análises de tendências dos acidentes ao longo do tempo."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["data_inversa"] = pd.to_datetime(self.df["data_inversa"])
        self.df["ano"] = self.df["data_inversa"].dt.year
        self.df["mes"] = self.df["data_inversa"].dt.month
        self.df["dia"] = self.df["data_inversa"].dt.day

    def get_yearly_trend(self) -> pd.DataFrame:
        """Tendência anual de acidentes."""
        return self.df.groupby("ano").agg({
            'id': 'count',
            'mortos': 'sum',
            'feridos': 'sum',
            'veiculos': 'sum'
        }).round(2)
    
    def get_monthly_trend(self) -> pd.DataFrame:
        """Tendência mensal de acidentes."""
        return self.df.groupby(["ano", "mes"]).agg({
            'id': 'count',
            'mortos': 'sum',
            'feridos': 'sum'
        }).unstack().fillna(0).round(2)
    
    def get_weekday_trend(self) -> pd.DataFrame:
        """Tendência por dia da semana."""
        return self.df.groupby("dia_semana").agg({
            'id': 'count',
            'mortos': 'sum',
            'feridos': 'sum'
        }).round(2)
    
    def get_severity_trend(self) -> pd.DataFrame:
        """Tendência da severidade dos acidentes ao longo dos anos."""
        self.df['indice_severidade'] = (
            self.df['mortos'] * 13 + self.df['feridos_graves'] * 5 +
            self.df['feridos_leves'] * 1
        ) / self.df['veiculos']
        
        return self.df.groupby("ano").agg({
            'indice_severidade': 'mean',
            'mortos': 'sum',
            'feridos_graves': 'sum',
            'feridos_leves': 'sum'
        }).round(2)
