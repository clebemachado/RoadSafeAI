import pandas as pd
from config.config_project import ConfigProject
from data_collection.collect_data import CollectData
from pathlib import Path
import shutil


from data_collection.file_download import FileDownloader

COLUMN_DESCRIPTION = "DESCRIPTION"
COLUMN_URL = "URL"
COLUMN_YEAR = "YEAR"
COLUMN_FILE_NAME = "FILE_NAME"

MESSAGE_ERROR_FILE_EXIST = '/files já existe. Não será feito novo download'
FILTER_TEXT_DOCUMENT_CSV_ACIDENTES = 'Documento CSV de Acidentes '
FILTER_TEXT_ALL_CASES = "Todas as causas e"
FILTER_GROUP_FOR_PEOPLE = "Agrupados por pessoa"

PATH_SAVE_FILE_YAML = "paths.save_files"
URL_REPLACE = "https://drive.usercontent.google.com/u/0/uc?id=ID_FILE&export=download"


class CollectDataDetran(CollectData):
    
    def __init__(self):
        super().__init__()
        self.config = ConfigProject()
        self.root_path = Path(__file__).parent.parent.parent
        self.type_file = ".csv"
        
    def execute(self, force_download=False):
        
        """
        Método principal para executar o fluxo de coleta e processamento de dados. Este método:
        1. Obtém os dados do Detran através de `__getDataFrame`.
        2. Limpa os dados utilizando `__clean_dataframe`.
        3. Gera os nomes dos arquivos com base nos dados limpos, utilizando `__generate_column_name_file`.
        4. Salva os arquivos localmente

        Parâmetros:
            force_execute: para excluir a pasta e baixar novamentes os dados.

        Retorno:
            Nenhum.
        """
        
        if self.__verify_if_file_folder_exist(force_download):
            print(f"Folder {self.root_path.stem}{MESSAGE_ERROR_FILE_EXIST}")
            return

        print("Iniciando coleta de dados")
        
        df = self.__getDataFrame()
        self.__clean_dataframe(df)
        
        df = self.__remove_rows_with_data_repeat(df) # Cuidado aqui
        
        self.__generate_column_name_file(df)
        self.__download_and_save_files(df)
        
        print("finalizando coleta de dados")
    
    
    def __verify_if_file_folder_exist(self, force_download):
        folder_files = self.root_path / "files" 
        
        if force_download and folder_files.exists() and folder_files.is_dir():
            """Se forçar eu removo a pasta e baixo novamente os arquivos"""
            shutil.rmtree(folder_files)
            return False
        
        return folder_files.is_dir()
    
    
    def __getDataFrame(self):
        """
        Obtém os dados em formato HTML de um URL configurado no projeto e converte para um DataFrame.
        Após a leitura dos dados, o método remove a primeira e a última linha (irrelevantes) e 
        define os nomes das colunas com base nas variáveis `COLUMN_DESCRIPTION` e `COLUMN_URL`.

        Parâmetros:
            Nenhum.

        Retorno:
            DataFrame: Retorna um DataFrame contendo os dados do Detran com as colunas renomeadas.
        """
        
        url = self.config.get("database.detran")
        
        df =  pd.read_html(url, extract_links="body")
        df = df[1] # Pega a primeira linha
        df = df.iloc[1:-1] #Exclui a primeira e a última linha
        df.columns = [COLUMN_DESCRIPTION, COLUMN_URL]
        return df
    
    
    def __clean_dataframe(self, df):
        """
        Limpa o DataFrame, realizando as seguintes operações:
        1. Extrai o segundo elemento de `COLUMN_URL`.
        2. Extrai o primeiro elemento de `COLUMN_DESCRIPTION`.
        3. Remove a string 'Documento CSV de Acidentes ' de `COLUMN_DESCRIPTION`.
        4. Extrai o ano (quatro dígitos) de `COLUMN_DESCRIPTION` e armazena em `COLUMN_YEAR`.
        5. Chama `__remove_rows_with_data_repeat` para remover dados repetidos ou não agrupados.

        Parâmetros:
            df (DataFrame): O DataFrame que será limpo.

        Retorno:
            Nenhum. O DataFrame `df` é modificado in-place.
        """
        

        df.loc[:, "URL"] = df["URL"].apply(lambda x: x[1].split("/")[-3])
        df.loc[:, "URL"] = df["URL"].apply(lambda x: URL_REPLACE.replace("ID_FILE", x))
        
        df.loc[:, COLUMN_DESCRIPTION] = df[COLUMN_DESCRIPTION].apply(lambda x: x[0])
        df.loc[:, COLUMN_DESCRIPTION] = df[COLUMN_DESCRIPTION].str.replace(FILTER_TEXT_DOCUMENT_CSV_ACIDENTES, '', regex=False)
        df[COLUMN_YEAR] = df[COLUMN_DESCRIPTION].str.extract(r'(\d{4})')
    
    def __remove_rows_with_data_repeat(self, df):
        """
        Remove linhas com dados repetidos ou não agrupados no DataFrame. Este método verifica se a 
        coluna `COLUMN_DESCRIPTION` contém a string 'Todas as causas e', e se encontrar, mantém apenas 
        essas linhas. Caso contrário, mantém os dados agrupados por ano.

        Parâmetros:
            df (DataFrame): O DataFrame a ser filtrado.

        Retorno:
            Nenhum. O DataFrame `df` é modificado in-place.
        """
        
        def filtrar_links(df_g):
            if df_g[COLUMN_DESCRIPTION].str.contains('Todas as causas e', na=False).any():
                return df_g[df_g[COLUMN_DESCRIPTION].str.contains('Todas as causas e', na=False)]
            else:
                return df_g
        return df.groupby(COLUMN_YEAR).apply(filtrar_links, include_groups=True).reset_index(drop=True)

        
    def __generate_column_name_file(self, df):
        """
        Gera nomes de arquivos com base nas descrições das colunas do DataFrame. Para cada valor em 
        `COLUMN_DESCRIPTION`, o método cria um nome de arquivo no formato de ano seguido de um sufixo 
        correspondente ao tipo de agrupamento.

        Parâmetros:
            df (DataFrame): O DataFrame contendo os dados com a coluna `COLUMN_DESCRIPTION`.

        Retorno:
            Nenhum. A coluna `COLUMN_FILE_NAME` é adicionada ao DataFrame `df`.
        """
        def get_name_file(col):
            """
            Função auxiliar que gera o nome do arquivo baseado no conteúdo de `COLUMN_DESCRIPTION`.
            
            Parâmetros:
                col (str): O valor da coluna `COLUMN_DESCRIPTION` para o qual o nome do arquivo será gerado.

            Retorno:
                str: O nome do arquivo gerado com base no ano e tipo de agrupamento.
            """
            year = col[:4]
        
            if FILTER_TEXT_ALL_CASES in col:
                return f"{year}_geral"
            elif FILTER_GROUP_FOR_PEOPLE in col:
                return f"{year}_agg_pessoa"
            else:
                return f"{year}_agg_ocorrencia"
        
        # Se não retornar esse valor, eu vou está criando uma nova referência de memória, pensar se é melhor adicionar no contexto da classe
        df.loc[:, COLUMN_FILE_NAME] = df[COLUMN_DESCRIPTION].apply(get_name_file)
    
    
    def __download_and_save_files(self, df):
        """
        Baixa os arquivos a partir dos links presentes no DataFrame e os salva com os nomes 
        gerados na coluna `COLUMN_FILE_NAME` (com a extensão `.csv`), utilizando a classe `FileDownloader`.

        Parâmetros:
            df (DataFrame): O DataFrame contendo os links e os nomes dos arquivos.

        Retorno:
            Nenhum.
        """
        for index, row in df.iterrows():
            
            url = row[COLUMN_URL]
            file_name = row[COLUMN_FILE_NAME]  
            
            # Todo: ver uma forma melhor de pegar o programa root ou como definir que antes de src é root
            download_folder = Path(__file__).parent.parent.parent / self.config.get(PATH_SAVE_FILE_YAML)
            FileDownloader.download_and_save(url, file_name, self.type_file, download_folder)