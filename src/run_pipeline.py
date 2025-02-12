#!/usr/bin/env python
# run_pipeline.py

import argparse
import logging

from model.non_tree_models import get_non_tree_models
from pipelines.model_pipeline import ModelingPipeline, create_tree_based_models
from pipelines.preprocessing_pipeline import PreprocessingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Execute pipeline de modelagem')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Seed para reprodutibilidade')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Número de estimadores para modelos ensemble')
    parser.add_argument('--output-dir', type=str, default='model_results',
                       help='Diretório para salvar resultados')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporção do conjunto de teste')
    parser.add_argument('--valid-size', type=float, default=0.2,
                       help='Proporção do conjunto de validação')
    parser.add_argument('--balance', type=str, default='smote',
                       choices=['smote', 'none'],
                       help='Estratégia de balanceamento')
    parser.add_argument('--dataset-type', type=str, default='base',
                       help='Tipo de dataset a ser usado')
    parser.add_argument('--collect-new-data', action='store_true',
                       help='Coletar novos dados')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Cria modelos
    tree_models = create_tree_based_models(
        random_state=args.random_state,
        n_estimators=args.n_estimators
    )
    non_tree_models = get_non_tree_models(random_state=args.random_state)
    all_models = tree_models + non_tree_models
    
    # Inicializa pipelines
    modeling_pipeline = ModelingPipeline(all_models, output_dir=args.output_dir)
    
    preprocessing = PreprocessingPipeline(
        collect_new_data=args.collect_new_data,
        dataset_type=args.dataset_type,
        test_size=args.test_size,
        valid_size=args.valid_size,
        balance_strategy=args.balance,
        random_state=args.random_state
    )
    
    # Executa pipeline
    logger.info("Iniciando processamento dos dados...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing.process_data()
    classes = sorted(y_train.unique())
    
    logger.info("Executando pipeline de modelagem...")
    results, comparison_df = modeling_pipeline.run_pipeline(
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        classes=classes
    )
    
    logger.info("\nComparação final dos modelos:")
    logger.info("\n" + str(comparison_df))

if __name__ == "__main__":
    main()