import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from uuid import uuid4

class CreateModelDecisionTree():
    def create(self):
        with st.expander("Configuração geral do DecisionTreeClassifier."):
            
            criterion = st.selectbox('Escolha o critério de divisão:', ['gini', 'entropy', 'log_loss'], index=0, key="criterion")
            splitter = st.selectbox('Escolha o método de divisão:', ['best', 'random'], index=0, key = "splitter")
            max_depth = st.slider('Escolha a profundidade máxima da árvore:', 1, 20, 10, step=1, key="max_depth")
            max_depth = None if max_depth == 20 else max_depth
            min_samples_split = st.slider('Mínimo de amostras para dividir um nó:', 2, 10, 2, key="min_samples_split")
            min_samples_leaf = st.slider('Mínimo de amostras para ser uma folha:', 1, 10, 1, key = "min_samples_leaf")
            min_weight_fraction_leaf = st.slider('Mínima fração de peso de amostra para folha:', 0.0, 0.5, 0.0, key="min_weight_fraction_leaf")
            max_features = st.selectbox('Escolha o valor de max_features:', ['None', 'sqrt', 'log2', 'auto'], index=0, key = "max_features")
            
            if max_features == 'None':
                max_features = None
                
            random_state = st.slider('Escolha o valor de random_state:', 0, 100, 42, key = "random_state_decision_tree")
            max_leaf_nodes = st.slider('Máximo de nós folha:', 1, 20, 10, step=1, key="max_leaf_nodes")
            max_leaf_nodes = None if max_leaf_nodes == 20 else max_leaf_nodes  
            min_impurity_decrease = st.slider('Mínima redução de impureza para dividir:', 0.0, 0.5, 0.0, key = "min_impurity_decrease")
            class_weight = st.selectbox('Escolha o peso da classe:', ['None', 'balanced'], index=0, key = "class_weight")
            
            if class_weight == 'None':
                class_weight = None
                
            ccp_alpha = st.slider('Valor de ccp_alpha para poda:', 0.0, 1.0, 0.0, key = "ccp_alpha")
            
            if st.button("Adicionar Modelo à Lista", key = "button_decision_tree"):
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    splitter=splitter,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    class_weight=class_weight,
                    ccp_alpha=ccp_alpha
                )
                """model = DecisionTreeClassifier(
                    random_state= 42,
                    class_weight='balanced'
                )"""
                if 'models' not in st.session_state:
                    st.session_state["models"] = {}
                    
                st.session_state["models"].update({"Decision Tree": model})
                st.success("Modelo adicionado à lista!")