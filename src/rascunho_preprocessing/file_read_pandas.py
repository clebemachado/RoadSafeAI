import pandas as pd
from  pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PandasReadFile:
    @staticmethod
    def read_csv_file(file_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, encoding='cp1252', sep=';')
        except Exception:
            logging.warning("Iniciando leitura dos arquivos com utf-8")
            return pd.read_csv(file_path, encoding='utf-8', sep=';')