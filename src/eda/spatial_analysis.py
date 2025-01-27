import pandas as pd
import numpy as np
from typing import Dict, List

class SpatialAnalysis:
    """Análises espaciais dos acidentes"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_state_stats(self) -> pd.DataFrame:
        """Estatisticas por estado"""
        return self.df.groupby('uf').agg({
            'id':'count',
            'mortos':['sum','mean'],
            'feridos':['sum','mean'],
            'veiculos':'sum'
        }).round(2)
    
    def get_highway_stats(self) -> pd.DataFrame:
        """Estatisticas por rodovia"""
        return self.df.groupby("br").agg({
            'id':'count',
            'mortos':['sum','mean'],
            'feridos':['sum','mean']
        }).sort_values(('id','count'), ascending=False).round(2)
    
    def get_accident_density(self) -> pd.DataFrame:
        """Densidade de acidentes por trecho."""
        return self.df.groupby(["br","km"]).size().reset_index(name='acidentes')