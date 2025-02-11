import os
import json
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config.inject_logger import inject_logger

from model.non_tree_model_evaluator import NonTreeModelEvaluator


@inject_logger
class NonTreeResultsSaver:
    """Classe específica para salvar resultados de modelos não baseados em árvores"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.non_tree_dir = os.path.join(output_dir, 'non_tree_models')
        os.makedirs(self.non_tree_dir, exist_ok=True)
    
    def save_metrics(self, metrics: Dict, model_name: str) -> None:
        """
        Salva métricas do modelo
        """
        model_dir = os.path.join(self.non_tree_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)
        
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"Métricas salvas em: {metrics_path}")
    
    def save_plots(self, model_name: str, evaluator: NonTreeModelEvaluator,
                  y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                  classes: List[str]) -> None:
        """
        Salva visualizações do modelo
        """
        model_dir = os.path.join(self.non_tree_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)
        
        # Matriz de Confusão
        evaluator.plot_confusion_matrix(y_true, y_pred, classes, 
                                      title=f'{model_name} - Confusion Matrix')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Curvas ROC
        if y_prob is not None:
            evaluator.plot_roc_curves(y_true, y_prob, classes)
            plt.savefig(os.path.join(model_dir, 'roc_curves.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        
        self.logger.info(f"Plots salvos em: {model_dir}")
    
    def save_model_comparison(self, all_results: Dict[str, Dict]) -> None:
        """
        Salva comparação entre todos os modelos (tree e non-tree)
        """
        # Prepara dados para comparação
        comparison_data = []
        for model_name, results in all_results.items():
            row = {
                'model': model_name,
                'type': 'Non-Tree' if model_name in ['Naive Bayes', 'KNN', 'Logistic Regression'] else 'Tree',
                'accuracy': results['test_metrics']['accuracy'],
                'precision': results['test_metrics']['precision'],
                'recall': results['test_metrics']['recall'],
                'f1': results['test_metrics']['f1']
            }
            comparison_data.append(row)
        
        # Salva comparação em CSV
        df_comparison = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(self.non_tree_dir, 'model_comparison.csv')
        df_comparison.to_csv(comparison_path, index=False)
        
        # Cria visualização da comparação
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_comparison, x='model', y=metric, hue='type')
            plt.title(f'Model Comparison - {metric.capitalize()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.non_tree_dir, f'comparison_{metric}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
        
        self.logger.info(f"Comparação dos modelos salva em: {self.non_tree_dir}")