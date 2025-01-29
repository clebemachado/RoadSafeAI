import pandas as pd

class DataFrameManipulation:
    
    @staticmethod
    def exctract_year_from_filename(filename: str) -> int:
        print(f"FileName: {filename}")
        return int(filename.split('datatran')[-1].split('.')[0])
    
    
    @staticmethod
    def convert_numeric(series: pd.Series) -> pd.Series:
        if series.dtype == 'object':
            series = series.astype(str).str.strip()
            series = series.str.replace(',','.')
        return pd.to_numeric(series, errors='coerce')