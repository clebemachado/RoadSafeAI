import calendar
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EDAAnalysis:
    """
    Classe responsável pela análise exploratória de dados do dataset de acidentes.
    """
    
    def __init__(self, output_dir: str = "graficos"):
        """
        Inicializa a classe EDA definindo o diretório de saída para os gráficos.
        
        Args:
            output_dir: Diretório onde os gráficos serão salvos
        """
        self.output_dir = Path(output_dir)
        self._setup_directories()
        
        # Configuração global do estilo dos gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _setup_directories(self):
        """Cria a estrutura de diretórios para salvar os gráficos."""
        directories = [
            'temporal', 'espacial', 'severidade','infraestrutura', 'estatistica'
        ]
        
        for dir_name in directories:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _save_plot(self, filename: str, category: str, dpi: int = 300):
        """
        Salva o gráfico atual no diretório apropriado.
        
        Args:
            filename: Nome do arquivo
            category: Categoria do gráfico (temporal, espacial, etc.)
            dpi: Resolução da imagem
        """
        plt.tight_layout()
        save_path = self.output_dir / category / f"{filename}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico salvo em: {save_path}")

    def analise_temporal(self, df: pd.DataFrame):
        """Realiza análises temporais dos acidentes."""
        logger.info("Iniciando análise temporal...")
        
        # Converter data_inversa para datetime
        df['data'] = pd.to_datetime(df['data_inversa'], format='%Y-%m-%d')
        
        # Distribuição por ano
        plt.figure(figsize=(12, 6))
        df['ano'] = df['data'].dt.year
        acidentes_por_ano = df['ano'].value_counts().sort_index()
        acidentes_por_ano.plot(kind='bar')
        plt.title('Distribuição de Acidentes por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Número de Acidentes')
        plt.xticks(rotation=45)
        self._save_plot('acidentes_por_ano', 'temporal')
        
        # Análise mensal
        plt.figure(figsize=(12, 6))
        df['mes'] = df['data'].dt.month
        acidentes_por_mes = df['mes'].value_counts().sort_index()
        acidentes_por_mes.index = [calendar.month_abbr[m] for m in acidentes_por_mes.index]
        acidentes_por_mes.plot(kind='bar')
        plt.title('Distribuição de Acidentes por Mês')
        plt.xlabel('Mês')
        plt.ylabel('Número de Acidentes')
        self._save_plot('acidentes_por_mes', 'temporal')
        
        # Análise por dia da semana
        plt.figure(figsize=(12, 6))
        df['dia_semana_num'] = df['data'].dt.dayofweek
        acidentes_por_dia = df['dia_semana_num'].value_counts().sort_index()
        acidentes_por_dia.index = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
        acidentes_por_dia.plot(kind='bar')
        plt.title('Distribuição de Acidentes por Dia da Semana')
        plt.xlabel('Dia da Semana')
        plt.ylabel('Número de Acidentes')
        self._save_plot('acidentes_por_dia', 'temporal')

    def analise_espacial(self, df: pd.DataFrame):
        """Realiza análises espaciais dos acidentes."""
        logger.info("Iniciando análise espacial...")
        
        # Distribuição por BR
        plt.figure(figsize=(12, 6))
        df['br'].value_counts().plot(kind='bar')
        plt.title('Distribuição de Acidentes por BR')
        plt.xlabel('BR')
        plt.ylabel('Número de Acidentes')
        plt.xticks(rotation=45)
        self._save_plot('acidentes_por_br', 'espacial')
        
        # Top 10 municípios
        plt.figure(figsize=(15, 8))
        df['municipio'].value_counts().head(10).plot(kind='barh')
        plt.title('Top 10 Municípios com Mais Acidentes')
        plt.xlabel('Número de Acidentes')
        plt.ylabel('Município')
        self._save_plot('top_10_municipios', 'espacial')

    def analise_severidade(self, df: pd.DataFrame):
      """Realiza análises de severidade dos acidentes."""
      logger.info("Iniciando análise de severidade...")
      
      # Tipos de acidentes vs número de vítimas
      plt.figure(figsize=(15, 8))
      
      # Cria o gráfico de barras empilhadas
      severidade_tipo = df.groupby('tipo_acidente')[['mortos', 'feridos_graves', 'feridos_leves']].mean()
      ax = severidade_tipo.plot(
          kind='bar', 
          stacked=True,
          color=['#FF9999', '#DAA520', '#90EE90'],  # Cores similares ao gráfico original
          width=0.8
      )
      
      # Configuração do título e labels
      plt.title('Média de Vítimas por Tipo de Acidente', pad=20, fontsize=12)
      plt.xlabel('Tipo de Acidente', labelpad=10)
      plt.ylabel('Número Médio de Vítimas', labelpad=10)
      
      # Ajuste das legendas
      plt.legend(
          ['mortos', 'feridos_graves', 'feridos_leves'],
          bbox_to_anchor=(1.02, 1),
          loc='upper left',
          borderaxespad=0,
          frameon=True
      )
      
      # Configuração da grade
      plt.grid(True, axis='y', alpha=0.3)
      
      # Ajuste dos rótulos do eixo x
      plt.xticks(
          rotation=45,
          ha='right',  # Alinhamento horizontal
          fontsize=8
      )
      
      # Ajuste das margens para evitar corte das legendas
      plt.tight_layout()
      
      # Salva o gráfico
      self._save_plot('vitimas_por_tipo', 'severidade', dpi=300)
      
      # Condições meteorológicas vs severidade
      plt.figure(figsize=(12, 6))
      severidade_clima = pd.crosstab(df['condicao_metereologica'], df['classificacao_acidente'])
      severidade_clima.plot(kind='bar', stacked=True)
      plt.title('Severidade por Condição Meteorológica')
      plt.xlabel('Condição Meteorológica')
      plt.ylabel('Número de Acidentes')
      plt.xticks(rotation=45)
      plt.tight_layout()
      self._save_plot('severidade_clima', 'severidade', dpi=300)

    def analise_infraestrutura(self, df: pd.DataFrame):
        """Realiza análises relacionadas à infraestrutura."""
        logger.info("Iniciando análise de infraestrutura...")
        
        # Tipo de pista vs acidentes
        plt.figure(figsize=(12, 6))
        df['tipo_pista'].value_counts().plot(kind='bar')
        plt.title('Distribuição de Acidentes por Tipo de Pista')
        plt.xlabel('Tipo de Pista')
        plt.ylabel('Número de Acidentes')
        plt.xticks(rotation=45)
        self._save_plot('acidentes_tipo_pista', 'infraestrutura')
        
        # Uso do solo
        plt.figure(figsize=(12, 6))
        df['uso_solo'].value_counts().plot(kind='bar')
        plt.title('Distribuição de Acidentes por Uso do Solo')
        plt.xlabel('Uso do Solo')
        plt.ylabel('Número de Acidentes')
        plt.xticks(rotation=45)
        self._save_plot('acidentes_uso_solo', 'infraestrutura')

    def analise_estatistica(self, df: pd.DataFrame):
        """Realiza análises estatísticas dos dados."""
        logger.info("Iniciando análise estatística...")
        
        colunas_boxplot = ['feridos_graves','feridos_leves','ignorados','ilesos','mortos','pessoas','veiculos']

        # Correlação entre variáveis numéricas
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[colunas_boxplot].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação - Variáveis Numéricas')
        self._save_plot('correlation_matrix', 'estatistica')
        
        
        # Boxplots para variáveis numéricas
        for col in colunas_boxplot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, y=col)
            plt.title(f'Distribuição de {col}')
            self._save_plot(f'boxplot_{col}', 'estatistica')

    def realizar_analise_completa(self, df: pd.DataFrame):
        """
        Realiza todas as análises exploratórias no dataset.
        
        Args:
            df: DataFrame com os dados dos acidentes
        """
        logger.info("Iniciando análise exploratória completa...")
        
        self.analise_temporal(df)
        self.analise_espacial(df)
        self.analise_severidade(df)
        self.analise_infraestrutura(df)
        self.analise_estatistica(df)
        
        logger.info("Análise exploratória completa finalizada!")
