import requests
import os
import zipfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileDownloader:
    @staticmethod
    def download_and_save(url: str, file_name: str, type: str = ".csv", download_folder: str = "files"):
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                
                content_type = response.headers.get("Content-Type", "").lower()
                
                extension = FileDownloader.get_extension(content_type)
                
                if not extension:
                    logger.info(f"Tipo de arquivo desconhecido: {content_type}")
                    return False
                
                file_name_with_extension = file_name + extension
                
                os.makedirs(download_folder, exist_ok=True) # Se existir ele não apaga
                file_path = os.path.join(download_folder, file_name_with_extension)
                
                with open(file_path, 'wb') as f:
                    logger.info(f"Arquivo salvo em: {file_path}")
                    f.write(response.content)
                
                FileDownloader.extract_if_file_is_zip(file_path)

                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Erro ao baixar ou salvar o arquivo {file_name}: {e}")
            return False
    
    @staticmethod
    def get_extension(content_type):
        return {
                'application/zip': '.zip',
                'text/csv': '.csv',
                'application/csv': '.csv',
                'application/octet-stream': '.zip'
        }.get(content_type, '')
    
    @staticmethod
    def extract_if_file_is_zip(file_path):
        if zipfile.is_zipfile(filename=file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".csv"):
                        csv_path = file_path.replace('.zip', '.csv')
                        with(open(csv_path, "wb")) as csv_file:
                            csv_file.write(zip_file.read(file))
                            logger.info(f"CSV extraido para: {csv_path}")
            os.remove(file_path)
        else:
            logger.error("Erro: O arquivo baixado não é um ZIP válido.")