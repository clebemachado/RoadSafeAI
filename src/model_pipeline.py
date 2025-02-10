import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator

from model.model_result_saver import ModelResultsSaver
from model.model_trainer import ModelTrainer
from pipeline import PreprocessingPipeline

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
        self.results_saver.setup_directories([t.name for t in self.trainers])
    
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
        Executa o pipeline completo de modelagem e salva resultado
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
            
            # Armazena resultados
            model_results = {
                'model': model,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
                'cv_results': {'mean': cv_mean, 'std': cv_std}
            }
            
            self.results[trainer.name] = model_results
            
            # Salva resultados do modelo
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
            
            # Obtém a função de plot de importância das features apenas se for um modelo baseado em árvore
            feature_importance_func = None
            if hasattr(model, 'feature_importances_'):
                feature_importance_func = trainer.tree_evaluator.plot_feature_importance
            
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
            
            self.results_saver.save_model_summary(
                model,
                trainer.name,
                model_results
            )
        
        # Salva resultados comparativos
        self.results_saver.save_comparison_results(
            self.results,
            self.compare_models
        )
        
        return self.results
    
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

# Exemplo de uso
def create_tree_based_models(
    random_state: int = 42,
    n_estimators: int = 100
) -> List[Tuple[str, BaseEstimator]]:
    """
    Cria uma lista de modelos baseados em árvore de decisão
    """
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.tree import DecisionTreeClassifier
    
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
            'Extra Trees',
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        ),
        (
            'Gradient Boosting',
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        )
    ]
    
    return models

def main():
    """Exemplo de como usar o pipeline de modelagem com modelos baseados em árvore"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Cria os modelos baseados em árvore
    models = create_tree_based_models(random_state=42, n_estimators=100)
    
    # Inicializa o pipeline
    pipeline = ModelingPipeline(models)
    
    pipeline = PreprocessingPipeline(
      collect_new_data=True,
      dataset_type='base',
      test_size=0.2,
      valid_size=0.2,
      balance_strategy='smote',
      random_state=42
    )

    X_train, X_valid, X_test, y_train, y_valid, y_test = pipeline.process_data()
    
    # Executa o pipeline (substitua com seus dados reais)
    results = pipeline.run_pipeline(
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        classes=['classe1', 'classe2', 'classe3']  # substitua com suas classes
    )

if __name__ == "__main__":
    main()