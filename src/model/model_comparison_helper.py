from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparisonHelper:
    """Classe para auxiliar na comparação entre diferentes tipos de modelos"""
    
    def compare_all_models(self, tree_results: Dict, non_tree_results: Dict) -> pd.DataFrame:
        """
        Compara todos os modelos e retorna um DataFrame com as métricas
        """
        all_results = {**tree_results, **non_tree_results}
        comparison_data = []
        
        for model_name, results in all_results.items():
            model_type = 'Non-Tree' if model_name in ['Naive Bayes', 'KNN', 'Logistic Regression'] else 'Tree'
            
            row = {
                'Model': model_name,
                'Type': model_type,
                'Accuracy': results['test_metrics']['accuracy'],
                'Precision': results['test_metrics']['precision'],
                'Recall': results['test_metrics']['recall'],
                'F1-Score': results['test_metrics']['f1']
            }
            
            if 'roc_auc_mean' in results['test_metrics']:
                row['ROC AUC'] = results['test_metrics']['roc_auc_mean']
                
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str) -> None:
        """
        Plota comparação visual entre modelos para uma métrica específica
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(data=comparison_df, x='Model', y=metric, hue='Type')
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()