import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class AnalisadorAcidentes:
    def __init__(self, df, pasta_saida='graficos'):
        """
        Inicializa o analisador com o DataFrame e cria as pastas necessárias
        """
        self.df = df.copy()
        self.pasta_base = pasta_saida
        self.criar_pastas()
        
        # Configurações globais para os gráficos
        plt.style.use(style='seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['font.size'] = 10
        
        # Converter data para datetime
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['ano'] = self.df['data'].dt.year
        self.df['mes'] = self.df['data'].dt.month

    def criar_pastas(self):
        """
        Cria a estrutura de pastas para organizar os gráficos
        """
        categorias = ['temporal', 'geografico', 'causalidade', 'severidade', 'estatistico', 'risco']
        for categoria in categorias:
            pasta = os.path.join(self.pasta_base, categoria)
            os.makedirs(pasta, exist_ok=True)

    def salvar_grafico(self, categoria, nome):
        """
        Salva o gráfico na pasta apropriada com resolução adequada
        """
        caminho = os.path.join(self.pasta_base, categoria, f"{nome}.png")
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()

    def analise_temporal(self):
        """
        Realiza análises temporais dos acidentes
        """
        # Evolução anual
        plt.figure(figsize=(14, 8))
        acidentes_por_ano = self.df['ano'].value_counts().sort_index()
        sns.lineplot(x=acidentes_por_ano.index, y=acidentes_por_ano.values)
        plt.title('Evolução do Número de Acidentes por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Número de Acidentes')
        plt.grid(True)
        self.salvar_grafico('temporal', 'evolucao_anual')

        # Distribuição por dia da semana
        plt.figure(figsize=(12, 8))
        ordem_dias = ['segunda-feira', 'terca-feira', 'quarta-feira', 'quinta-feira', 
                     'sexta-feira', 'sabado', 'domingo']
        sns.countplot(data=self.df, y='dia_semana', order=ordem_dias)
        plt.title('Distribuição de Acidentes por Dia da Semana')
        plt.xlabel('Número de Acidentes')
        plt.ylabel('Dia da Semana')
        self.salvar_grafico('temporal', 'distribuicao_dia_semana')

        # Heatmap período x gravidade
        plt.figure(figsize=(12, 8))
        periodo_gravidade = pd.crosstab(self.df['periodo_dia'], self.df['gravidade_acidente'])
        sns.heatmap(periodo_gravidade, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Relação entre Período do Dia e Gravidade dos Acidentes')
        self.salvar_grafico('temporal', 'heatmap_periodo_gravidade')

    def analise_geografica(self):
        """
        Realiza análises geográficas dos acidentes
        """
        # Top 10 municípios com mais acidentes
        plt.figure(figsize=(14, 8))
        top_municipios = self.df['municipio'].value_counts().head(10)
        sns.barplot(x=top_municipios.values, y=top_municipios.index)
        plt.title('Top 10 Municípios com Maior Número de Acidentes')
        plt.xlabel('Número de Acidentes')
        self.salvar_grafico('geografico', 'top_municipios')

        # Distribuição por tipo de pista
        plt.figure(figsize=(12, 8))
        gravidade_pista = pd.crosstab(self.df['tipo_pista'], self.df['gravidade_acidente'], 
                                    normalize='index') * 100
        gravidade_pista.plot(kind='bar', stacked=True)
        plt.title('Distribuição da Gravidade dos Acidentes por Tipo de Pista')
        plt.xlabel('Tipo de Pista')
        plt.ylabel('Porcentagem')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self.salvar_grafico('geografico', 'distribuicao_tipo_pista')

    def analise_causalidade(self):
        """
        Realiza análises de causalidade dos acidentes
        """
        # Top causas por gravidade
        plt.figure(figsize=(14, 8))
        top_causas = self.df['causa_acidente'].value_counts().head(10).index
        causa_grave = self.df[self.df['causa_acidente'].isin(top_causas)]
        sns.countplot(data=causa_grave, y='causa_acidente', hue='gravidade_acidente')
        plt.title('Top 10 Causas de Acidentes por Gravidade')
        plt.xlabel('Número de Acidentes')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self.salvar_grafico('causalidade', 'top_causas_gravidade')

    def analise_severidade(self):
        """
        Realiza análises de severidade dos acidentes
        """
        # Taxa de mortalidade por tipo de acidente
        plt.figure(figsize=(14, 8))
        mortalidade = self.df.groupby('tipo_acidente').agg({
            'mortos': 'sum',
            'tipo_acidente': 'count'
        }).rename(columns={'tipo_acidente': 'total'})
        mortalidade['taxa_mortalidade'] = (mortalidade['mortos'] / mortalidade['total']) * 100
        
        sns.barplot(x=mortalidade['taxa_mortalidade'], 
                   y=mortalidade.index)
        plt.title('Taxa de Mortalidade por Tipo de Acidente (%)')
        plt.xlabel('Taxa de Mortalidade')
        self.salvar_grafico('severidade', 'taxa_mortalidade')

        # Relação entre número de veículos e gravidade
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df, x='gravidade_acidente', y='veiculos')
        plt.title('Distribuição do Número de Veículos por Gravidade do Acidente')
        plt.xlabel('Gravidade do Acidente')
        plt.ylabel('Número de Veículos')
        self.salvar_grafico('severidade', 'veiculos_gravidade')

    def analise_risco(self):
        """
        Realiza análises de risco dos acidentes
        """
        # Índice de periculosidade por BR
        plt.figure(figsize=(14, 8))
        risco_br = self.df.groupby('br').agg({
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
        self.salvar_grafico('risco', 'indice_periculosidade_br')

        # Relação uso do solo x gravidade
        plt.figure(figsize=(12, 8))
        uso_solo_grav = pd.crosstab(self.df['uso_solo'], 
                                   self.df['gravidade_acidente'], 
                                   normalize='index') * 100
        uso_solo_grav.plot(kind='bar', stacked=True)
        plt.title('Distribuição da Gravidade dos Acidentes por Uso do Solo')
        plt.xlabel('Uso do Solo')
        plt.ylabel('Porcentagem')
        plt.legend(title='Gravidade', bbox_to_anchor=(1.05, 1))
        self.salvar_grafico('risco', 'uso_solo_gravidade')

    def executar_todas_analises(self):
        """
        Executa todas as análises disponíveis
        """
        print("Iniciando análises...")
        self.analise_temporal()
        print("Análises temporais concluídas")
        self.analise_geografica()
        print("Análises geográficas concluídas")
        self.analise_causalidade()
        print("Análises de causalidade concluídas")
        self.analise_severidade()
        print("Análises de severidade concluídas")
        self.analise_risco()
        print("Análises de risco concluídas")
        print("\nTodas as análises foram concluídas com sucesso!")

# Exemplo de uso:
if __name__ == "__main__":
    # Carregar os dados
    df = pd.read_csv('df_ma_features.csv')
    
    # Criar o analisador
    analisador = AnalisadorAcidentes(df)
    
    # Executar todas as análises
    analisador.executar_todas_analises()