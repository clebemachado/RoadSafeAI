from typing import Dict, Literal, Optional, Tuple
from config.inject_logger import inject_logger
from sklearn.pipeline import Pipeline
from preprocessing.data_balancing_06 import DataBalance
from preprocessing.data_split_05 import DataSplit
from preprocessing.transformers import (
    DataCleaningTransformer, DataCollectionTransformer, DataEncodingTransformer,
    DataStandardizeTransformer, DatasetMergerTransformer, FeatureEngineeringTransformer
)
import pandas as pd

@inject_logger
class PreprocessingPipeline:
    """
    Classe principal responsável por gerenciar o pipeline completo de pré-processamento,
    desde a coleta de dados até o pré-processamento final
    """
    def __init__(
        self,
        collect_new_data: bool = True,
        dataset_type: Literal['base', 'complete'] = 'base',
        test_size: float = 0.2,
        valid_size: float = 0.2,
        balance_strategy: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Inicializa o pipeline de pré-processamento
        """
        self.collect_new_data = collect_new_data
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.valid_size = valid_size
        self.balance_strategy = balance_strategy
        self.random_state = random_state
        
        steps = []
        
        self.logger.info(f"COLLECT NEW DATA: {self.collect_new_data}")
        
        if self.collect_new_data:
            steps.extend([
                ('collect_data', DataCollectionTransformer()),
                ('merge_datasets', DatasetMergerTransformer(dataset_type=self.dataset_type))
            ])
        
        steps.extend([
            ('cleaning', DataCleaningTransformer()),
            ('standardize', DataStandardizeTransformer()),
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('encoding', DataEncodingTransformer())
        ])
        
        self.pipeline = Pipeline(steps)
        
        self.data_splitter = DataSplit()
        self.data_balancer = DataBalance() if balance_strategy else None

    def process_data(
        self,
        input_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Processa os dados através do pipeline completo
        """
        self.logger.info(f"Iniciando pipeline completo de pré-processamento (usando dataset {self.dataset_type})...")
        
        df_processed = self.pipeline.fit_transform(input_data if input_data is not None else pd.DataFrame())
        self.logger.info(f"Dimensões após pré-processamento: {df_processed.shape}")
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_splitter.prepare_data(
            df_processed,
            test_size=self.test_size,
            valid_size=self.valid_size,
            random_state=self.random_state
        )
        
        # Garantir que todos os tipos numéricos sejam float64 antes do balanceamento
        numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[numeric_columns] = X_train[numeric_columns].astype('float64')
        X_valid[numeric_columns] = X_valid[numeric_columns].astype('float64')
        X_test[numeric_columns] = X_test[numeric_columns].astype('float64')
        
        self.logger.info("Tipos de dados após conversão:")
        for col in X_train.columns:
            self.logger.info(f"- {col}: {X_train[col].dtype}")
        
        if self.balance_strategy:
            self.logger.info(f"Aplicando estratégia de balanceamento {self.balance_strategy}...")
            try:
                X_train, y_train = self.data_balancer.balance_data(
                    X_train,
                    y_train,
                    strategy=self.balance_strategy,
                    random_state=self.random_state
                )
            except Exception as e:
                self.logger.error(f"Erro durante o balanceamento: {str(e)}")
                self.logger.error("Tipos de dados após tentativa de balanceamento:")
                for col in X_train.columns:
                    self.logger.error(f"- {col}: {X_train[col].dtype}")
                raise
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def get_feature_names(self) -> Dict[str, list]:
        """
        Obtém os nomes das features após pré-processamento
        """
        encoder = self.pipeline.named_steps['encoding'].encoder
        return encoder.get_feature_names()