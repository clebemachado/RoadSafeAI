import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AnaliseExploratoria:
    def __init__(self, output_dir='graficos'):
        """
        Inicializa a classe de análise exploratória.
        """
        self.output_dir = output_dir
        self._criar_diretorios()
        
    def __del__(self):
        """
        Cleanup method to ensure all plots are properly closed
        """
        plt.close('all')
        
    def _criar_diretorios(self):
        """Cria o diretório para salvar os gráficos se não existir"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _salvar_grafico(self, nome):
        """Salva o gráfico atual no diretório especificado"""
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{nome}.png'))
        plt.close()
        
    def evolucao_anual(self, df):
        """Gera gráfico de evolução anual dos acidentes"""
        plt.figure(figsize=(14, 8))
        acidentes_por_ano = df['ano'].value_counts().sort_index()
        sns.lineplot(x=acidentes_por_ano.index, y=acidentes_por_ano.values)
        plt.title('Evolução do Número de Acidentes por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Número de Acidentes')
        plt.grid(True)
        self._salvar_grafico('evolucao_anual')
        
    def distribuicao_dia_semana(self, df):
        """Gera gráfico de distribuição por dia da semana"""
        plt.figure(figsize=(12, 8))
        ordem_dias = ['segunda-feira', 'terca-feira', 'quarta-feira', 'quinta-feira', 
                     'sexta-feira', 'sabado', 'domingo']
        sns.countplot(data=df, y='dia_semana', order=ordem_dias)
        plt.title('Distribuição de Acidentes por Dia da Semana')
        plt.xlabel('Número de Acidentes')
        plt.ylabel('Dia da Semana')
        self._salvar_grafico('distribuicao_dia_semana')
        
    def heatmap_periodo_gravidade(self, df):
        """Gera heatmap de período x gravidade"""
        plt.figure(figsize=(12, 8))
        periodo_gravidade = pd.crosstab(df['periodo_dia'], df['gravidade_acidente'])
        sns.heatmap(periodo_gravidade, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Relação entre Período do Dia e Gravidade dos Acidentes')
        self._salvar_grafico('heatmap_periodo_gravidade')
        
    def top_municipios(self, df):
        """Gera gráfico dos top 10 municípios"""
        plt.figure(figsize=(14, 8))
        top_municipios = df['municipio'].value_counts().head(10)
        sns.barplot(x=top_municipios.values, y=top_municipios.index)
        plt.title('Top 10 Municípios com Maior Número de Acidentes')
        plt.xlabel('Número de Acidentes')
        self._salvar_grafico('top_municipios')
        
    def distribuicao_tipo_pista(self, df):
        """Gera gráfico de distribuição por tipo de pista"""
        plt.figure(figsize=(12, 8))
        gravidade_pista = pd.crosstab(df['tipo_pista'], df['gravidade_acidente'], 
                                     normalize='index') * 100
        gravidade_pista.plot(kind='bar', stacked=True)
        plt.title('Distribuição da Gravidade dos Acidentes por Tipo de Pista')
        plt.xlabel('Tipo de Pista')
        plt.ylabel('Porcentagem')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self._salvar_grafico('distribuicao_tipo_pista')
        
    def taxa_mortalidade(self, df):
        """Gera gráfico de taxa de mortalidade por tipo de acidente"""
        plt.figure(figsize=(14, 8))
        mortalidade = df.groupby('tipo_acidente').agg({
            'mortos': 'sum',
            'tipo_acidente': 'count'
        }).rename(columns={'tipo_acidente': 'total'})
        mortalidade['taxa_mortalidade'] = (mortalidade['mortos'] / mortalidade['total']) * 100
        sns.barplot(x=mortalidade['taxa_mortalidade'], y=mortalidade.index)
        plt.title('Taxa de Mortalidade por Tipo de Acidente (%)')
        plt.xlabel('Taxa de Mortalidade')
        self._salvar_grafico('taxa_mortalidade')
        
    def indice_periculosidade_br(self, df):
        """Gera gráfico de índice de periculosidade por BR"""
        plt.figure(figsize=(14, 8))
        br_mask = ~df['br'].isin([0, 1])
        df_br_valida = df[br_mask]
        risco_br = df_br_valida.groupby('br').agg({
            'mortos': 'sum',
            'feridos_graves': 'sum',
            'feridos_leves': 'sum',
            'br': 'count'
        }).rename(columns={'br': 'total_acidentes'})
        
        risco_br['indice_periculosidade'] = (
            (risco_br['mortos'] * 13) +
            (risco_br['feridos_graves'] * 5) +
            risco_br['feridos_leves']
        ) / risco_br['total_acidentes']
        
        top_br_risco = risco_br.sort_values('indice_periculosidade', ascending=False).head(10)
        sns.barplot(x=top_br_risco.index, y=top_br_risco['indice_periculosidade'])
        plt.title('Índice de Periculosidade por BR (Top 10)')
        plt.xlabel('BR')
        plt.ylabel('Índice de Periculosidade')
        plt.xticks(rotation=45)
        self._salvar_grafico('indice_periculosidade_br')
        
    def relacao_uso_solo_gravidade(self, df):
        """Gera gráfico de relação entre uso do solo e gravidade"""
        plt.figure(figsize=(12, 8))
        uso_solo_grav = pd.crosstab(df['uso_solo'], df['gravidade_acidente'], 
                                   normalize='index') * 100
        uso_solo_grav.plot(kind='bar', stacked=True)
        plt.title('Distribuição da Gravidade dos Acidentes por Uso do Solo')
        plt.xlabel('Uso do Solo')
        plt.ylabel('Porcentagem')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self._salvar_grafico('relacao_uso_solo_gravidade')
        
    def top_causas_gravidade(self, df):
        """Gera gráfico de top causas por gravidade"""
        plt.figure(figsize=(14, 8))
        top_causas = df['causa_acidente_grupo'].value_counts().head(10).index
        causa_grave = df[df['causa_acidente_grupo'].isin(top_causas)]
        sns.countplot(data=causa_grave, y='causa_acidente_grupo', hue='gravidade_acidente')
        plt.title('Top 10 Causas de Acidentes por Gravidade')
        plt.xlabel('Número de Acidentes')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self._salvar_grafico('top_causas_gravidade')
        
    def gerar_todas_analises(self, df):
        """
        Executa todas as análises exploratórias e salva os gráficos.
        """
        self.evolucao_anual(df)
        self.distribuicao_dia_semana(df)
        self.heatmap_periodo_gravidade(df)
        self.top_municipios(df)
        self.distribuicao_tipo_pista(df)
        self.taxa_mortalidade(df)
        self.indice_periculosidade_br(df)
        self.relacao_uso_solo_gravidade(df)
        self.top_causas_gravidade(df)
        
        print(f"Todos os gráficos foram salvos no diretório: {self.output_dir}")

