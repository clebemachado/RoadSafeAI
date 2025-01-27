from typing import Dict, List

import numpy as np
import pandas as pd


class TemporalAnalysis:
    """ Analise temporais dos acidentes."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["data_inversa"] = pd.to_datetime(self.df["data_inversa"])
        self.df["hora"] = self.df["data_inversa"].dt.hour
        self.df["mes"] = self.df["data_inversa"].dt.month
        self.df["ano"] = self.df["data_inversa"].dt.year

    def get_yearly_stats(self) -> pd.DataFrame:
        """ Estatisticas anuais de acidentes """
        return self.df.groupby("ano").agg({
            'id':'count',
            'mortos':'sum',
            'feridos':'sum',
            'veiculos':'sum'
        }).round(2)
    
    def get_monthly_pattern(self) -> pd.DataFrame:
        """ Padrões mensais de acidentes """
        return self.df.groupby('mes').agg({
            'id':'count',
            'mortos':'mean',
            'feridos':'mean'
        }).round(2)
    
    def get_hourly_pattern(self) -> pd.DataFrame:
        """ Padrões horários de acidentes. """
        return self.df.groupby('hora').agg({
            'id':'count',
            'mortos':'sum',
            'feridos':'sum'
        }).round(2)
    
    def get_weekday_pattern(self) -> pd.DataFrame:
        """ Padrões por dia da semana """
        return self.df.groupby('dia_semana').agg({
            'id':'count',
            'mortos':'mean',
            'feridos':'mean'
        }).round(2)
    
    