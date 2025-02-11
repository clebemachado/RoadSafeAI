import logging
from functools import wraps

def inject_logger(cls):
    """
    Decorador para injetar um logger configurado em uma classe.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    cls.logger = logging.getLogger(cls.__module__ + "." + cls.__name__)
    
    return cls