import json
import logging
import os
from datetime import datetime
from typing import Dict

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pipelines.preprocessing_pipeline import PreprocessingPipeline

class ModelOptimizer:
    """Classe responsável pela otimização de hiperparâmetros"""
    
    def __init__(self, output_dir: str = "optimization_results"):
        """
        Inicializa o otimizador de modelos
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Configuração do logger
        self.logger = logging.getLogger(__name__)
        
        # Dicionário para armazenar os estudos do Optuna
        self.studies = {}
        
    def optimize_decision_tree(self, X: pd.DataFrame, y: pd.Series, 
                             n_trials: int = 100) -> Dict:
        """
        Otimiza hiperparâmetros para Decision Tree
        """
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': 'balanced'
            }
            
            model = DecisionTreeClassifier(**params, random_state=42)
            score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return score.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.studies['Decision Tree'] = study
        return study.best_params
        
    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                             n_trials: int = 100) -> Dict:
        """
        Otimiza hiperparâmetros para Random Forest
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', 
                                                       ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params, random_state=42)
            score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return score.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.studies['Random Forest'] = study
        return study.best_params
    
    def optimize_catboost(self, X: pd.DataFrame, y: pd.Series, 
                         n_trials: int = 100) -> Dict:
        """
        Otimiza hiperparâmetros para CatBoost
        """
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'thread_count': -1,
                'verbose': False
            }
            
            model = CatBoostClassifier(**params, random_state=42)
            score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return score.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.studies['CatBoost'] = study
        return study.best_params
    
    def save_optimization_results(self, model_name: str, best_params: Dict, 
                                study: optuna.Study) -> None:
        """
        Salva os resultados da otimização
        """
        model_dir = os.path.join(self.run_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)
        
        params_path = os.path.join(model_dir, 'best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(model_dir, 'optimization_history.csv')
        trials_df.to_csv(trials_path, index=False)
        
        stats = {
            'best_value': study.best_value,
            'best_iteration': study.best_trial.number,
            'n_trials': len(study.trials),
            'datetime_start': study.trials[0].datetime_start.isoformat(),
            'datetime_complete': study.trials[-1].datetime_complete.isoformat()
        }
        
        stats_path = os.path.join(model_dir, 'optimization_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
        self.logger.info(f"Resultados da otimização salvos em: {model_dir}")
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          n_trials: int = 100) -> Dict[str, Dict]:
        """
        Otimiza todos os modelos baseados em árvore
        """
        optimization_results = {}
        
        self.logger.info("Otimizando Decision Tree...")
        dt_params = self.optimize_decision_tree(X, y, n_trials)
        self.save_optimization_results('Decision Tree', dt_params, 
                                     self.studies['Decision Tree'])
        optimization_results['Decision Tree'] = dt_params
        
        self.logger.info("Otimizando Random Forest...")
        rf_params = self.optimize_random_forest(X, y, n_trials)
        self.save_optimization_results('Random Forest', rf_params, 
                                     self.studies['Random Forest'])
        optimization_results['Random Forest'] = rf_params
        
        self.logger.info("Otimizando CatBoost...")
        cb_params = self.optimize_catboost(X, y, n_trials)
        self.save_optimization_results('CatBoost', cb_params, 
                                     self.studies['CatBoost'])
        optimization_results['CatBoost'] = cb_params
        
        self.save_comparison_summary(optimization_results)
        
        return optimization_results
    
    def save_comparison_summary(self, optimization_results: Dict[str, Dict]) -> None:
        """
        Salva um resumo comparativo dos resultados da otimização
        """
        summary = {
            model_name: {
                'best_params': params,
                'best_score': self.studies[model_name].best_value,
                'n_trials': len(self.studies[model_name].trials)
            }
            for model_name, params in optimization_results.items()
        }
        
        summary_path = os.path.join(self.run_dir, 'optimization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Cria um DataFrame comparativo
        comparison_data = []
        for model_name, results in summary.items():
            comparison_data.append({
                'Model': model_name,
                'Best F1 Score': results['best_score'],
                'Number of Trials': results['n_trials']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(self.run_dir, 'models_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"Resumo comparativo salvo em: {self.run_dir}")

def main():
    """Exemplo de uso do otimizador"""
    
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
    
    # Inicializa o otimizador
    optimizer = ModelOptimizer()
    
    # Otimiza todos os modelos
    best_params = optimizer.optimize_all_models(X_train, y_train, n_trials=100)
    
    return best_params

if __name__ == "__main__":
    main()