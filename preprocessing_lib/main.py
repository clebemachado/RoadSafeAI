import pandas as pd
from preprocessing_lib.cleaning import clean_data
from preprocessing_lib.preprocessing import preprocess_data
from preprocessing_lib.exploration import plot_distribution
from preprocessing_lib.outlier_detection import OutlierDetection
from preprocessing_lib.feature_transformation import FeatureTransformation
from preprocessing_lib.normalization import Normalization
from preprocessing_lib.feature_selection import FeatureSelection
from preprocessing_lib.feature_engineering import FeatureEngineering
from preprocessing_lib.missing_values import MissingValues

# Carregar os dados
df = pd.read_csv("datatran2024.csv")

# Aplicar limpeza e pré-processamento
df = clean_data(df)
df = preprocess_data(df)

# Tratamento de valores ausentes
missing_handler = MissingValues(df)
missing_handler.fill_missing_with_median("coluna_numerica")
df = missing_handler.get_data()

# Tratamento de outliers
outlier_handler = OutlierDetection(df)
outlier_handler.remove_outliers_iqr("coluna_numerica")
df = outlier_handler.get_data()

# Transformação de variáveis
transformer = FeatureTransformation(df)
transformer.log_transform("coluna_numerica")
df = transformer.get_transformed_data()

# Normalização dos dados
normalizer = Normalization(df)
normalizer.min_max_scale(["coluna_numerica", "coluna2"])
df = normalizer.get_scaled_data()

# Engenharia de atributos
feature_eng = FeatureEngineering(df)
feature_eng.extract_date_features("data_inversa")
df = feature_eng.get_data()

# Exploração dos dados
plot_distribution(df, "tipo_acidente")
