import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(df, column):
    """Gera um gráfico de distribuição para a coluna especificada."""
    sns.histplot(df[column], kde=True)
    plt.show()
