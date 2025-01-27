import pandas as pd
import numpy as np
from typing import Dict, List

class SeverityAnalysis:
    """Análises de severidade dos acidentes."""

    def __init__(self, df:pd.DataFrame):
        self.df =df.copy()
        self._calculate_severity_index()

    def _calculate_severity_index(self):
        """Calcular indice de severidade."""
        self.df['indice_severidade'] = (
            self.df['mortos'] * 13 + self.df['feridos_graves'] * 5 +
            self.df['feridos_leves'] * 1
        ) / self.df['veiculos']

    def get_severity_by_cause(self) -> pd.DataFrame:
        """Severidade por causa de acidente"""
        return self.df.groupby('causa_acidente').agg({
            'indice_severidade': 'mean',
            'mortos':['sum','mean'],
            'feridos_graves':['sum','mean'],
            'feridos_leves':['sum','mean']
        }).round(2)

    def get_severity_by_tipe(self) -> pd.DataFrame:
        """Severidade por tipo de acidente"""
        return self.df.groupby('tipo_acidente').agg({
            'indice_severidade':'mean',
            'mortos':['sum','mean'],
            'feridos':['sum','mean']
        }).round(2)