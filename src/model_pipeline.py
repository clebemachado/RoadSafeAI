import logging
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from pipeline import PreprocessingPipeline
from model.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelingPipeline:
    """Classe principal para gerenciar o pipeline de modelagem"""
    
    def __init__(self, models: List[Tuple[str, BaseEstimator]]):
        """
        Inicializa o pipeline de modelagem
        
        Args:
            models: Lista de tuplas (nome_modelo, modelo)
        """
        self.trainers = [ModelTrainer(model, name) for name, model in models]
        self.results = {}
    
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
        Executa o pipeline completo de modelagem
        
        Args:
            X_train: Features de treino
            X_valid: Features de validação
            X_test: Features de teste
            y_train: Target de treino
            y_valid: Target de validação
            y_test: Target de teste
            classes: Lista com os nomes das classes
            
        Returns:
            Dict com os resultados de todos os modelos
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
            
            # Plota matriz de confusão
            y_pred = model.predict(X_test)
            trainer.evaluator.plot_confusion_matrix(y_test, y_pred, classes)
            
            # Armazena resultados
            self.results[trainer.name] = {
                'model': model,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
                'cv_results': {'mean': cv_mean, 'std': cv_std}
            }
        
        return self.results
    
    def compare_models(self, metric: str = 'f1') -> None:
        """
        Compara os modelos usando uma métrica específica
        
        Args:
            metric: Nome da métrica para comparação
        """
        comparison = {
            'Treino': [r['train_metrics'][metric] for r in self.results.values()],
            'Validação': [r['valid_metrics'][metric] for r in self.results.values()],
            'Teste': [r['test_metrics'][metric] for r in self.results.values()]
        }
        
        df_comparison = pd.DataFrame(comparison, index=self.results.keys())
        
        plt.figure(figsize=(12, 6))
        df_comparison.plot(kind='bar')
        plt.title(f'Comparação dos Modelos - Métrica: {metric}')
        plt.xlabel('Modelo')
        plt.ylabel(f'Score {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Exemplo de uso
def create_tree_based_models(
    random_state: int = 42,
    n_estimators: int = 100
) -> List[Tuple[str, BaseEstimator]]:
    """
    Cria uma lista de modelos baseados em árvore de decisão
    
    Args:
        random_state: Semente aleatória
        n_estimators: Número de estimadores para modelos ensemble
        
    Returns:
        Lista de tuplas (nome_modelo, modelo)
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier
    )
    
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
    
    # Compara os modelos
    pipeline.compare_models(metric='f1')

if __name__ == "__main__":
    main()