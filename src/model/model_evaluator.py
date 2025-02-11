import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from config.inject_logger import inject_logger

@inject_logger
class ModelEvaluator:
    """Classe responsável pela avaliação dos modelos"""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Não foi possível calcular ROC AUC: {str(e)}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> None:
        cm = confusion_matrix(y_true, y_pred)
        
        # Normaliza a matriz
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Cria o heatmap
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Matriz de Confusão')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()