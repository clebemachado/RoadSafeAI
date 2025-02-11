import logging

from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran
from src.data_collection.merge_datasets import DatasetMerger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        collect_data : CollectData = CollectDataDetran()
        collect_data.execute() # Colocar como false caso já tenha dados
        
        dataset_merge: DatasetMerger = DatasetMerger();
        dataset_merge.execute()
    except Exception as e:
        logger.error(f"Erro durante a unificação dos datasets: {str(e)}")

if __name__ == "__main__":
    main()
    