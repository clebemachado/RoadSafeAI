import yaml
from pathlib import Path

class ConfigProject:
    _instance = None
    
    def __new__(cls, config_path = "config.yaml"):
        # O new é chamado antes do INIT
        if cls._instance is None:
            cls._instance = super(ConfigProject, cls).__new__(cls)
            base_path = Path(__file__).parent.parent.parent / config_path

            cls._instance.__load_config(base_path)
        return cls._instance
        
            
    def __load_config(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            