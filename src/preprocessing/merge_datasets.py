import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.config_project import ConfigProject
from preprocessing.dataframe_manipulation import DataFrameManipulation
from preprocessing.file_read_pandas import PandasReadFile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetMerger:
    """Classe para unificar datasets de acidentes de diferentes anos,"""
    
    def __init__(self):
        self.config:ConfigProject = ConfigProject()
        
        self.project_root: Optional[Path] = self._get_project_root()
        self.data_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        
        self.base_columns = [
            'id', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km',
            'municipio', 'causa_acidente', 'tipo_acidente', 'classificacao_acidente',
            'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista',
            'tracado_via', 'uso_solo', 'pessoas', 'mortos', 'feridos_leves',
            'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos'
        ]

        self.extra_columns = ['latitude', 'longitude', 'regional', 'delegacia', 'uop']
        self._config_paths()
    
    
    def execute(self):
        merged_df_base = self._merge_datasets(start_year = 2007, end_year = 2024, include_extra_columns=False)
        self.save_merged_dataset(merged_df_base, "datatran_merged_base_2007_2024.csv")
        
        
        merged_def_complete = self._merge_datasets(start_year = 2017, end_year = 2024, include_extra_columns=False)
        self.save_merged_dataset(merged_def_complete, "datatran_merged_complete_2017_2024.csv")
        
    
    def _get_project_root(self) -> Path:
        """Obtém o diretório raiz do projeto baseado no local do arquivo atual."""
        return Path(__file__).resolve().parent.parent.parent
    
    
    def _config_paths(self):
        """Configura os diretórios de entrada e saída do projeto."""
        try:
            save_path = self.config.get("paths.save_files")
            output_path = self.config.get("paths.output_files")

            if save_path is None or output_path is None:
                raise ValueError("Os caminhos 'save_files' e 'output_files' não estão definidos na configuração.")

            self.data_dir = self.project_root / save_path
            self.output_dir = self.project_root / output_path

            self._ensure_directories_exist()

        except Exception as e:
            raise RuntimeError(f"Erro ao configurar caminhos: {e}")
    
    
    def _ensure_directories_exist(self):
        """Cria diretórios se eles não existirem e verifica a existência do diretório de dados."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Diretório de dados não encontrado: {self.data_dir}")
    
    
    def _merge_datasets(self, start_year:int, end_year:int, include_extra_columns: bool) -> pd.DataFrame:
        
        csv_files = sorted([
            f for f in self.data_dir.glob('datatran*.csv')
            if start_year <= DataFrameManipulation.exctract_year_from_filename(f.name) <= end_year
        ])
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado para o periódo {start_year}-{end_year}")
        
        logger.info(f"{len(csv_files)} arquivos CSV encontrados para o período {start_year}-{end_year}.")
        
        df_list = self._find_all_csvs(include_extra_columns, csv_files)

        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df['data_inversa'] = pd.to_datetime(merged_df['data_inversa'], format="mixed")
        merged_df = merged_df.sort_values('data_inversa')
    
        if not include_extra_columns:
            merged_df = merged_df.drop(columns=self.extra_columns, errors="ignore")

        return merged_df
    
    
    def _find_all_csvs(self, include_extra_columns: bool, csv_files: List[Path]) -> List[pd.DataFrame]:
        df_list = []
        
        for file_path in csv_files:
            year = DataFrameManipulation.exctract_year_from_filename(file_path.name)
            logger.info(f"Processando dataset do ano {year}: {file_path.name}")
            
            try:
                df = PandasReadFile.read_csv_file(file_path)
                df_processed = self.process_dataset(df, year, include_extra_columns)
                df_list.append(df_processed)
            except Exception as e:
                logger.error(f"Erro ao processar {file_path.name}: {e}")
        
        return df_list
    
    
    def process_dataset(self, df: pd.DataFrame, year:int, include_extra_columns: bool = False) -> pd.DataFrame:
        if 'ano' not in df.columns:
            df['ano'] = year
    
        all_columns  = self.base_columns + (self.extra_columns if include_extra_columns else [])
        df = df.reindex(columns=all_columns, fill_value=None)

        numeric_columns = {
            'br': 'Int64',
            'km': 'float64',
            'pessoas': 'Int64',
            'mortos': 'Int64',
            'feridos_leves': 'Int64',
            'feridos_graves': 'Int64',
            'ilesos': 'Int64',
            'ignorados': 'Int64',
            'feridos': 'Int64',
            'veiculos': 'Int64'
        }
    
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = DataFrameManipulation.convert_numeric(df[col]).astype(dtype)
        
        if include_extra_columns and {'latitude', 'longitude'}.issubset(df.columns):
            df['latitude'] = pd.to_numeric(df['latitude'].str.replace(',', '.'), errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'].str.replace(',', '.'), errors='coerce')

        return df
    
    
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
        