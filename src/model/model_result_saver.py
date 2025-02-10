import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelResultsSaver:
  """Classe responsável por salvar todos os resultados dos modelos"""
  
  def __init__(self, output_dir: str = "model_results"):
    self.output_dir = output_dir
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.run_dir = os.path.join(output_dir, self.timestamp)

  def setup_directories(self, model_names: List[str]) -> None:
    os.makedirs(self.run_dir, exist_ok=True)
    
    for model_name in model_names:
      model_dir = os.path.join(self.run_dir, model_name.replace(" ","_"))
      os.makedirs(model_dir, exist_ok=True)
      
  def save_metrics(self, metrics: Dict, model_name: str, filename: str) -> None:
    model_dir = os.path.join(self.run_dir, model_name.replace(" ","_"))
    filepath = os.path.join(model_dir,filename)
    
    with open(filepath, 'w') as f:
      json.dump(metrics, f, indent=4)
      
    logger.info(f"Metricas salvas em: {filepath}")
    
  def save_plots(self, model: BaseEstimator, model_name: str, 
                  y_test: pd.Series, y_pred: np.ndarray,
                  classes: List[str], X_test: pd.DataFrame,
                  confusion_matrix_func, feature_importance_func) -> None:
        """
        Salva visualizações do modelo
        """
        model_dir = self.get_model_dir(model_name)
        
        # Matriz de confusão
        plt.figure(figsize=(10, 8))
        confusion_matrix_func(y_test, y_pred, classes)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()
        
        # Feature importance para modelos baseados em árvore
        if hasattr(model, 'feature_importances_') and feature_importance_func:
            plt.figure(figsize=(12, 8))
            importances = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            })
            importances = importances.sort_values('importance', ascending=False).head(20)
            
            sns.barplot(data=importances, x='importance', y='feature')
            plt.title(f'Top 20 Features Mais Importantes - {model_name}')
            plt.xlabel('Importância')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            plt.savefig(os.path.join(model_dir, 'feature_importance.png'), 
                       bbox_inches='tight', 
                       dpi=300)
            plt.close()
        
        logger.info(f"Plots salvos em: {model_dir}")
        
  def get_model_dir(self, model_name: str) -> str:
    """
    Retorna o diretório específico do modelo, criando-o se necessário
    
    Args:
        model_name: Nome do modelo
        
    Returns:
        Caminho do diretório do modelo
    """
    # Simplifica a sanitização - apenas substitui espaços por underscores
    safe_name = model_name.replace(" ", "_")
    model_dir = os.path.join(self.run_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir
    
  def save_model_summary(self, model: BaseEstimator, model_name: str, 
                          model_results: Dict) -> None:
    """
    Salva resumo detalhado do modelo
    """
    model_dir = os.path.join(self.run_dir, model_name.replace(" ", "_"))
    filepath = os.path.join(model_dir, 'model_summary.txt')
    
    with open(filepath, 'w') as f:
      f.write(f"Model: {model_name}\n")
      f.write("=" * 50 + "\n\n")
      
      # Parâmetros do modelo
      f.write("Model Parameters:\n")
      f.write("-" * 20 + "\n")
      params = model.get_params()
      for param, value in params.items():
          f.write(f"{param}: {value}\n")
      f.write("\n")
      
      # Métricas de performance
      for dataset in ['train', 'valid', 'test']:
          f.write(f"{dataset.capitalize()} Metrics:\n")
          f.write("-" * 20 + "\n")
          metrics = model_results[f'{dataset}_metrics']
          for metric, value in metrics.items():
              f.write(f"{metric}: {value:.4f}\n")
          f.write("\n")
      
      # Resultados da validação cruzada
      f.write("Cross-validation Results:\n")
      f.write("-" * 20 + "\n")
      f.write(f"Mean: {model_results['cv_results']['mean']:.4f}\n")
      f.write(f"Std: {model_results['cv_results']['std']:.4f}\n")
    
    logger.info(f"Resumo do modelo salvo em: {filepath}")
    
    
  def save_comparison_results(self, results: Dict[str, Dict], 
                              comparison_func) -> None:
    """
    Salva resultados comparativos de todos os modelos
    """
    # Cria DataFrame com todas as métricas
    comparison_data = []
    for model_name, model_results in results.items():
        for dataset in ['train', 'valid', 'test']:
            metrics = model_results[f'{dataset}_metrics']
            comparison_data.append({
                'model': model_name,
                'dataset': dataset,
                **metrics
            })
    
    # Salva CSV com todas as métricas
    df_comparison = pd.DataFrame(comparison_data)
    csv_path = os.path.join(self.run_dir, 'model_comparison.csv')
    df_comparison.to_csv(csv_path, index=False)
    
    # Gera e salva gráficos comparativos para cada métrica
    metrics = [col for col in df_comparison.columns 
              if col not in ['model', 'dataset']]
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        comparison_func(metric=metric)
        plt.savefig(os.path.join(self.run_dir, f'comparison_{metric}.png'))
        plt.close()
    
    logger.info(f"Resultados comparativos salvos em: {self.run_dir}")
    
  def get_run_directory(self) -> str:
    """Retorna o diretório da execução atual"""
    return self.run_dir