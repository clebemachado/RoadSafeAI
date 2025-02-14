import pandas as pd
import logging

logging.basicConfig(filename="cleaning.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def clean_data(df):
    """ Limpeza de dados: remoção de duplicatas e valores inconsistentes """
    
    initial_shape = df.shape
    df = df.drop_duplicates()
    
    # Remover linhas com valores inconsistentes na coluna 'tipo_acidente'
    df = df[df['tipo_acidente'].notna()]
    
    logging.info(f"Removidas {initial_shape[0] - df.shape[0]} linhas duplicadas ou inconsistentes.")
    
    return df
