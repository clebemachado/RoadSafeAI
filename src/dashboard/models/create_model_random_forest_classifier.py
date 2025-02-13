import streamlit as st
from sklearn.ensemble import RandomForestClassifier
        
class CreateModelRandomForestClassifier:
    
    def create(self):
        with st.expander("Configuração geral do RandomForestClassifier."):
        
            n_estimators = st.slider('Número de estimadores:', 1, 200, 100)
            criterion = st.selectbox('Escolha o critério de divisão:', ['gini', 'entropy'], index=0)
            max_depth = st.slider('Escolha a profundidade máxima da árvore:', 1, 20, 10, step=1)
            max_depth = None if max_depth == 20 else max_depth
            min_samples_split = st.slider('Mínimo de amostras para dividir um nó:', 2, 10, 2)
            min_samples_leaf = st.slider('Mínimo de amostras para ser uma folha:', 1, 10, 1)
            min_weight_fraction_leaf = st.slider('Mínima fração de peso de amostra para folha:', 0.0, 0.5, 0.0)
            max_features = st.selectbox('Escolha o valor de max_features:', ['auto', 'sqrt', 'log2', None], index=1)
            if max_features == 'None':
                max_features = None
            max_leaf_nodes = st.slider('Máximo de nós folha:', 1, 20, 10, step=1)
            max_leaf_nodes = None if max_leaf_nodes == 20 else max_leaf_nodes
            min_impurity_decrease = st.slider('Mínima redução de impureza para dividir:', 0.0, 0.5, 0.0)
            bootstrap = st.checkbox('Usar Bootstrap:', value=True)
            oob_score = st.checkbox('Calcular o OOB Score:', value=False)
            n_jobs = st.number_input('Número de jobs para paralelização:', value=-1)
            random_state = st.number_input('Escolha o valor de random_state:', value=42)
            verbose = st.slider('Nível de verbosidade:', 0, 10, 0)
            warm_start = st.checkbox('Usar warm start:', value=False)
            class_weight = st.selectbox('Escolha o peso da classe:', ['None', 'balanced'], index=0)
            if class_weight == 'None':
                class_weight = None
            ccp_alpha = st.slider('Valor de ccp_alpha para poda:', 0.0, 1.0, 0.0)
            max_samples = st.slider('Máximo de amostras para treinamento:', 1, 1000, 1000, step=10)
            max_samples = None if max_samples == 1000 else max_samples

            if st.button("Adicionar Modelo à Lista"):
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=verbose,
                    warm_start=warm_start,
                    class_weight=class_weight,
                    ccp_alpha=ccp_alpha,
                    max_samples=max_samples
                )
                """model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )"""
                if 'models' not in st.session_state:
                    st.session_state["models"] = {}
                    
                st.session_state["models"].update({"Random Forest": model})
                st.success("Modelo adicionado à lista!")
