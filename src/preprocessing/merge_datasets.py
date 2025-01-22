import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetMerger:
  """Classe para unificar datasets de acidentes de diferentes anos,"""
  
  def __init__(self, project_root: str = Path(__file__), data_dir: str = "data/raw"):
    
    
    current_file = Path(__file__)
    current_project_root = current_file.parent.parent.parent
    
    self.data_dir = current_project_root / "data" / "raw"
    self.output_dir = current_project_root / "data" / "processed"
    self.project_root = current_project_root
    
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not self.data_dir.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {self.data_dir}")
    
    self.base_columns = [
        'id', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km',
        'municipio', 'causa_acidente', 'tipo_acidente', 'classificacao_acidente',
        'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista',
        'tracado_via', 'uso_solo', 'pessoas', 'mortos', 'feridos_leves',
        'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos'
    ]
    
    self.extra_columns = [
        'latitude', 'longitude', 'regional', 'delegacia', 'uop'
    ]
    
  def read_csv_file(self, file_path: Path) -> pd.DataFrame:
    try:
      print(file_path)
      df = pd.read_csv(file_path, encoding='cp1252', sep=';')
    except UnicodeDecodeError:
      df = pd.read_csv(file_path, encoding='utf-8', sep=';')
    
    return df


  def exctract_year_from_filename(self, filename: str) -> int:
    return int(filename.split('datatran')[-1].split('.')[0])
  
  def process_dataset(self, df: pd.DataFrame, year:int) -> pd.DataFrame:
    if 'ano' not in df.columns:
      df['ano'] = year
      
    for col in self.extra_columns:
      if col not in df.columns:
        df[col] = None
        
    for col in self.base_columns:
      if col not in df.columns:
        logger.warning(f"Coluna {col} não encontrada no dataset de {year}")
        df[col] = None
        
    numeric_columns = ['br', 'km', 'pessoas', 'mortos', 'feridos_leves','feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos']
    
    for col in numeric_columns:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if 'latitude' in df.columns and 'longitude' in df.columns:
      df['latitude'] = pd.to_numeric(df['latitude'].str.replace(',','.'), errors='coerce')
      df['longitude'] = pd.to_numeric(df['longitude'].str.replace(',','.'), errors='coerce')
      
    return df
  
  def merge_datasets(self) -> pd.DataFrame:
    dfs = []
    
    csv_files = sorted([f for f in self.data_dir.glob('datatran*.csv')])
    
    for file_path in csv_files:
      year = self.exctract_year_from_filename(file_path.name)
      logger.info(f"Processando dataset do ano {year}")
      
      df = self.read_csv_file(file_path)
      df_processed = self.process_dataset(df, year)
      dfs.append(df_processed)
      
    merged_df = pd.concat(dfs, ignore_index=True)
    
    merged_df['data_inversa'] = pd.to_datetime(merged_df['data_inversa'], format="mixed")
    merged_df = merged_df.sort_values('data_inversa')
    
    return merged_df
  
  def save_merged_dataset(self, df: pd.DataFrame, filename: str = "datatran_merged.csv"):
    output_path = self.output_dir / filename
    
    df.to_csv(output_path, index=False, encoding="utf-8-sig", sep=';')
    logger.info(f"Dataset unificado salvo em {output_path}")
    
    metadata = {
      'total_registros': len(df),
      'periodo': f"{df['data_inversa'].min():%Y-%m-%d} a {df['data_inversa'].max():%Y-%m-%d}",
      'ultima_atualizacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      'colunas': list(df.columns)
    }
    
    metadata_path = self.output_dir / 'metadata.txt'
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
      for key, value in metadata.items():
        f.write(f"{key}: {value}\n")
        
    logger.info(f"Metadados salvos em {metadata_path}")

    
def main():
  try:
    merger = DatasetMerger()
    merged_df = merger.merge_datasets()
    merger.save_merged_dataset(merged_df)
  except Exception as e:
    logger.error(f"Erro durante a unificação dos datasets: {str(e)}")
    raise

if __name__ == "__main__":
    main()
    