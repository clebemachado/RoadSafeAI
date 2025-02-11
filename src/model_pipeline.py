# model_pipeline.py
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator

from model.model_comparison_helper import ModelComparisonHelper
from model.model_result_saver import ModelResultsSaver
from model.model_trainer import ModelTrainer
from model.non_tree_model_evaluator import NonTreeModelEvaluator
from model.non_tree_model_saver import NonTreeResultsSaver
from pipeline import PreprocessingPipeline
from model.non_tree_models import get_non_tree_models


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelingPipeline:
    """Classe principal para gerenciar o pipeline de modelagem"""
    
    def __init__(self, models: List[Tuple[str, BaseEstimator]], 
                 output_dir: str = "model_results"):
        """
        Inicializa o pipeline de modelagem
        
        Args:
            models: Lista de tuplas (nome_modelo, modelo)
            output_dir: Diretório para salvar os resultados
        """
        self.trainers = [ModelTrainer(model, name) for name, model in models]
        self.results = {}
        self.results_saver = ModelResultsSaver(output_dir)
        self.non_tree_evaluator = NonTreeModelEvaluator()
        self.non_tree_saver = NonTreeResultsSaver(output_dir)
        self.comparison_helper = ModelComparisonHelper()
        self.results_saver.setup_directories([t.name for t in self.trainers])
        
        # Lista de modelos não baseados em árvore
        self.non_tree_models = ['Naive Bayes', 'KNN', 'Logistic Regression']
    
    def run_pipeline(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        y_test: pd.Series,
        classes: List[str]
    ) -> Dict:
        """
        Executa o pipeline completo de modelagem e salva resultados
        """
        for trainer in self.trainers:
            logger.info(f"\nProcessando modelo: {trainer.name}")
            
            # Treina o modelo
            model = trainer.train(X_train, y_train)
            
            # Avalia nos conjuntos de dados
            train_metrics = trainer.evaluate(X_train, y_train, "treino")
            valid_metrics = trainer.evaluate(X_valid, y_valid, "validação")
            test_metrics = trainer.evaluate(X_test, y_test, "teste")
            
            # Realiza validação cruzada
            cv_mean, cv_std = trainer.cross_validate(X_train, y_train)
            
            # Gera predições para plots
            y_pred = model.predict(X_test)
            
            try:
                y_prob = model.predict_proba(X_test)
            except (AttributeError, NotImplementedError):
                y_prob = None
            
            # Armazena resultados básicos
            model_results = {
                'model': model,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
                'cv_results': {'mean': cv_mean, 'std': cv_std}
            }
            
            # Processamento específico para modelos não baseados em árvore
            if trainer.name in self.non_tree_models:
                non_tree_metrics = self.non_tree_evaluator.calculate_metrics(
                    y_test, y_pred, y_prob, trainer.name
                )
                model_results['non_tree_metrics'] = non_tree_metrics
                
                # Salva métricas e plots específicos
                self.non_tree_saver.save_metrics(non_tree_metrics, trainer.name)
                self.non_tree_saver.save_plots(
                    trainer.name,
                    self.non_tree_evaluator,
                    y_test,
                    y_pred,
                    y_prob,
                    classes
                )
            # Processamento para modelos baseados em árvore
            else:
                # Salva métricas gerais
                self.results_saver.save_metrics(
                    {
                        'train_metrics': train_metrics,
                        'valid_metrics': valid_metrics,
                        'test_metrics': test_metrics,
                        'cv_results': {'mean': cv_mean, 'std': cv_std}
                    },
                    trainer.name,
                    'metrics.json'
                )
                
                # Obtém função de plot de importância das features
                feature_importance_func = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance_func = trainer.tree_evaluator.plot_feature_importance
                elif hasattr(model, 'get_feature_importance'):  # Para CatBoost
                    feature_importance_func = lambda model, feature_names, top_n: \
                        trainer.tree_evaluator.plot_feature_importance(
                            model, 
                            feature_names, 
                            top_n,
                            importance_type='feature_importances_'
                        )
                
                # Salva plots específicos de árvore
                self.results_saver.save_plots(
                    model,
                    trainer.name,
                    y_test,
                    y_pred,
                    classes,
                    X_test,
                    trainer.evaluator.plot_confusion_matrix,
                    feature_importance_func
                )
            
            # Salva resumo do modelo
            self.results_saver.save_model_summary(
                model,
                trainer.name,
                model_results
            )
            
            self.results[trainer.name] = model_results
        
        # Comparação final entre todos os modelos
        comparison_df = self.comparison_helper.compare_all_models(
            {name: results for name, results in self.results.items() 
             if name not in self.non_tree_models},
            {name: results for name, results in self.results.items() 
             if name in self.non_tree_models}
        )
        
        # Salva comparação final
        self.non_tree_saver.save_model_comparison(self.results)
        self.results_saver.save_comparison_results(
            self.results,
            self.compare_models
        )
        
        return self.results, comparison_df
    
    def compare_models(self, metric: str = 'f1') -> None:
        """
        Compara os modelos usando uma métrica específica
        """
        comparison = {
            'Treino': [r['train_metrics'][metric] for r in self.results.values()],
            'Validação': [r['valid_metrics'][metric] for r in self.results.values()],
            'Teste': [r['test_metrics'][metric] for r in self.results.values()]
        }
        
        df_comparison = pd.DataFrame(comparison, index=self.results.keys())
        df_comparison.plot(kind='bar')
        plt.title(f'Comparação dos Modelos - Métrica: {metric}')
        plt.xlabel('Modelo')
        plt.ylabel(f'Score {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()

def create_tree_based_models(
    random_state: int = 42,
    n_estimators: int = 100
) -> List[Tuple[str, BaseEstimator]]:
    """
    Cria uma lista de modelos baseados em árvore de decisão
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from catboost import CatBoostClassifier

    
    models = [
        (
            'Decision Tree',
            DecisionTreeClassifier(
                random_state=random_state,
                class_weight='balanced'
            )
        ),
        (
            'Random Forest',
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        ),
        (
            'CatBoost',
            CatBoostClassifier(
                iterations=n_estimators,
                random_seed=random_state,
                verbose=False,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                thread_count=-1
            )
        )
    ]
    
    return models

def main():
    """Exemplo de como usar o pipeline de modelagem com todos os modelos"""

    # Cria os modelos baseados em árvore
    tree_models = create_tree_based_models(random_state=42, n_estimators=100)
    
    # Obtém os modelos não baseados em árvore
    non_tree_models = get_non_tree_models(random_state=42)
    
    # Combina todos os modelos
    all_models = tree_models + non_tree_models
    
    # Inicializa o pipeline
    pipeline = ModelingPipeline(all_models)
    
    # Inicializa o pipeline de pré-processamento
    preprocessing = PreprocessingPipeline(
        collect_new_data=True,
        dataset_type='base',
        test_size=0.2,
        valid_size=0.2,
        balance_strategy='smote',
        random_state=42
    )

    # Processa os dados
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing.process_data()
    
    # Define as classes com base nos dados reais
    classes = sorted(y_train.unique())
    
    # Executa o pipeline
    results, comparison_df = pipeline.run_pipeline(
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        classes=classes
    )
    
    logger.info("\nComparação final dos modelos:")
    logger.info("\n" + str(comparison_df))

if __name__ == "__main__":
    main()