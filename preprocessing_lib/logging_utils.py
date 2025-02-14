import logging

logging.basicConfig(filename="preprocessing.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def log_step(step_name, description):
    """ Registra um passo do processamento """
    
    logging.info(f"{step_name}: {description}")
