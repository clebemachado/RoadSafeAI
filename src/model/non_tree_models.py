# non_tree_models.py
from typing import List, Tuple

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_non_tree_models(random_state: int = 42) -> List[Tuple[str, BaseEstimator]]:
    """
    Cria uma lista de modelos que não são baseados em árvores de decisão
    
    Args:
        random_state: Semente para reprodutibilidade
        
    Returns:
        Lista de tuplas (nome_modelo, modelo)
    """
    models = [
        (
            'Naive Bayes',
            GaussianNB(
                var_smoothing=1e-9  # Porção da variância máxima para adicionar a variâncias
            )
        ),
        (
            'KNN',
            KNeighborsClassifier(
                n_neighbors=5,  # Número de vizinhos
                weights='distance',  # Peso baseado na distância
                metric='minkowski',  # Métrica de distância
                n_jobs=-1  # Usar todos os cores disponíveis
            )
        ),
        (
            'Logistic Regression',
            LogisticRegression(
                multi_class='multinomial',  # Para classificação multiclasse
                solver='lbfgs',  # Algoritmo de otimização
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
        )
    ]
    
    return models