import pandas as pd
from preprocessing_lib.cleaning import clean_data
from preprocessing_lib.preprocessing import preprocess_data
from preprocessing_lib.exploration import plot_distribution

# Carregar os dados
df = pd.read_csv("datatran2024.csv")

# Aplicar limpeza e pré-processamento
df = clean_data(df)
df = preprocess_data(df)

# Exploração dos dados
plot_distribution(df, "tipo_acidente")
