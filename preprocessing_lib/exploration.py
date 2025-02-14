import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Novo

def plot_distribution(df, column):
    """ Gera um gráfico de distribuição para uma coluna """
    
    fig = px.histogram(df, x=column, title=f'Distribuição de {column}', nbins=20)  # Novo gráfico interativo
    fig.show()
