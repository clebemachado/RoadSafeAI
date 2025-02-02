from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AccidentVisualizer:
    """Classe para visualização detalhada dos dados de acidentes."""
    
    def __init__(self):
        # Configuração do estilo
        sns.set_palette("husl")
    
    def plot_time_series(self, data: pd.Series, title: str, xlabel: str, ylabel: str,
                        figsize: Tuple[int, int] = (15, 6)) -> None:
        """Plotagem de séries temporais (ex.: acidentes ao longo do tempo)."""
        plt.figure(figsize=figsize)
        data.plot(kind='line', marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
    
    def plot_bar_chart(self, data: pd.Series, title: str, xlabel: str, ylabel: str,
                       horizontal: bool = False, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plotagem de gráficos de barras (ex.: número de acidentes por tipo, causa, etc.)."""
        plt.figure(figsize=figsize)
        kind = 'barh' if horizontal else 'bar'
        data.plot(kind=kind)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
    
    def plot_heatmap(self, data: pd.DataFrame, title: str, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plotagem de mapas de calor para correlações entre variáveis ou dados espaciais."""
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title(title)
        plt.tight_layout()
    
    def plot_box_plot(self, data: pd.DataFrame, x: str, y: str, title: str, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plotagem de box plots (ex.: variabilidade de severidade por tipo de acidente)."""
        plt.figure(figsize=figsize)
        sns.boxplot(data=data, x=x, y=y)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    def plot_scatter(self, data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                     title: str = '', figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plotagem de gráficos de dispersão para visualizar relações entre variáveis numéricas ou categóricas."""
        plt.figure(figsize=figsize)
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.tight_layout()
    
    def plot_distribution(self, data: pd.Series, title: str, xlabel: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plotagem de distribuições de variáveis numéricas (ex.: distribuição de severidade)."""
        plt.figure(figsize=figsize)
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequência')
        plt.tight_layout()

    # Função para exibir a severidade por tipo de acidente
    def plot_severity_by_type(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Exibe a severidade média por tipo de acidente."""
        severity_by_type = df.groupby('tipo_acidente').agg({'indice_severidade': 'mean'}).sort_values('indice_severidade', ascending=False)
        self.plot_bar_chart(severity_by_type['indice_severidade'], 
                            title='Severidade Média por Tipo de Acidente',
                            xlabel='Tipo de Acidente',
                            ylabel='Severidade Média',
                            figsize=figsize)
    
    # Função para exibir a distribuição de acidentes por mês/ano
    def plot_monthly_accidents(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 6)) -> None:
        """Exibe o número de acidentes por mês/ano."""
        monthly_accidents = df.groupby(['ano', 'mes']).size().unstack().fillna(0)
        monthly_accidents.plot(kind='line', figsize=figsize, marker='o')
        plt.title('Número de Acidentes por Mês ao Longo dos Anos')
        plt.xlabel('Ano')
        plt.ylabel('Número de Acidentes')
        plt.grid(True)
        plt.tight_layout()

    # Função para visualização de acidentes por condição meteorológica
    def plot_weather_impact(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Exibe o impacto das condições meteorológicas na severidade dos acidentes."""
        weather_severity = df.groupby('condicao_metereologica').agg({'indice_severidade': 'mean'}).sort_values('indice_severidade', ascending=False)
        self.plot_bar_chart(weather_severity['indice_severidade'], 
                            title='Impacto das Condições Meteorológicas na Severidade dos Acidentes',
                            xlabel='Condição Meteorológica',
                            ylabel='Severidade Média',
                            figsize=figsize)

    # Função para visualização de acidentes por rodovia
    def plot_highway_accidents(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Exibe o número de acidentes por rodovia (BR)."""
        highway_accidents = df.groupby('br').size().sort_values(ascending=False).head(10)
        self.plot_bar_chart(highway_accidents, 
                            title='Top 10 Rodovias com Maior Número de Acidentes',
                            xlabel='Rodovia',
                            ylabel='Número de Acidentes',
                            figsize=figsize)

    # Função para visualização de densidade de acidentes por trecho
    def plot_accident_density(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Exibe a densidade de acidentes por trecho de rodovia."""
        accident_density = df.groupby(['br', 'km']).size().reset_index(name='acidentes')
        self.plot_scatter(accident_density, 
                          x='br', 
                          y='km', 
                          hue='acidentes', 
                          title='Densidade de Acidentes por Trecho',
                          figsize=figsize)
