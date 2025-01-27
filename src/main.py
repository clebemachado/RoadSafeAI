from config.config_project import ConfigProject
from data_collection.collect_data import CollectData
from data_collection.collect_data_detran import CollectDataDetran

import sys
sys.path.append('src')

config = ConfigProject()

collect_data : CollectData = CollectDataDetran()
collect_data.execute(force_execute=config.get("config.download_files")) # Colocar como false caso já tenha dados
