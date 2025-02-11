import logging
from preprocessing.pipelines.preprocessing_pipeline import PreprocessingPipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Exemplo de uso do pipeline"""
    try:
        pipeline = PreprocessingPipeline(
            collect_new_data=True,
            dataset_type='base',
            test_size=0.2,
            valid_size=0.2,
            balance_strategy='smote',
            random_state=42
        )
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = pipeline.process_data()
        
        feature_names = pipeline.get_feature_names()
        
        logger.info("\nPré-processamento finalizado com sucesso!")
        logger.info(f"Formato dados de treino: {X_train.shape}")
        logger.info(f"Formato dados de validação: {X_valid.shape}")
        logger.info(f"Formato dados de teste: {X_test.shape}")
        logger.info("\nFeatures:")
        for feature_type, features in feature_names.items():
            logger.info(f"- {feature_type}: {len(features)} features")
        
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()