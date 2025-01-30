from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AccidentVisualizer:
    """Classe para visualização dos dados de acidentes."""
    
    def __init__(self):
        # Configuração do estilo
        sns.set_palette("husl")
    
    def plot_time_series(self, data: pd.Series, title: str,
                        xlabel: str, ylabel: str,
                        figsize: Tuple[int, int] = (15, 6)) -> None:
        """Plotagem de séries temporais."""
        plt.figure(figsize=figsize)
        data.plot(kind='line', marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
    
    def plot_bar_chart(self, data: pd.Series, title: str,
                      xlabel: str, ylabel: str,
                      horizontal: bool = False,
                      figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plotagem de gráficos de barras."""
        plt.figure(figsize=figsize)
        kind = 'barh' if horizontal else 'bar'
        data.plot(kind=kind)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
    
    def plot_heatmap(self, data: pd.DataFrame, title: str,
                    figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plotagem de mapas de calor."""
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title(title)
        plt.tight_layout()
    
    def plot_box_plot(self, data: pd.DataFrame, x: str, y: str,
                     title: str, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plotagem de box plots."""
        plt.figure(figsize=figsize)
        sns.boxplot(data=data, x=x, y=y)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    def plot_scatter(self, data: pd.DataFrame, x: str, y: str,
                    hue: Optional[str] = None, title: str = '',
                    figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plotagem de gráficos de dispersão."""
        plt.figure(figsize=figsize)
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.tight_layout()
    
    def plot_distribution(self, data: pd.Series, title: str,
                        xlabel: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plotagem de distribuições."""
        plt.figure(figsize=figsize)
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequência')
        plt.tight_layout()
        
        