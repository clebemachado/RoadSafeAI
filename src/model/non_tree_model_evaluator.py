import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NonTreeModelEvaluator:
    """Classe específica para avaliação de modelos não baseados em árvores"""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray, model_name: str) -> Dict:
        """
        Calcula métricas específicas para modelos não baseados em árvores
        """
        # Métricas básicas
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Calcula ROC AUC para cada classe (one-vs-rest)
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        
        if y_prob is not None:
            metrics['roc_auc'] = {}
            for i, class_name in enumerate(lb.classes_):
                if y_prob.shape[1] > i:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    metrics['roc_auc'][str(class_name)] = auc(fpr, tpr)
            
            metrics['roc_auc_mean'] = np.mean(list(metrics['roc_auc'].values()))
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            classes: List[str], title: str = 'Confusion Matrix') -> None:
        """
        Plota a matriz de confusão
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       classes: List[str]) -> None:
        """
        Plota as curvas ROC para cada classe
        """
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(classes):
            if y_prob.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()