from catboost import CatBoostClassifier
import streamlit as st

class CreateModelCatBoost():
    def create(self):
        with st.expander("Configuração geral do CatBoostClassifier."):
            
            iterations = st.slider('Número de iterações:', 100, 5000, 100, step=100, key="catboost_iterations")
            random_seed = st.number_input('Escolha o valor de random_seed:', value=42, key="catboost_random_seed")
            learning_rate = st.slider('Taxa de aprendizado:', 0.001, 1.0, 0.1, step=0.001, key="catboost_learning_rate")
            depth = st.slider('Profundidade da árvore:', 1, 16, 6, step=1, key="catboost_depth")
            l2_leaf_reg = st.slider('Regularização L2:', 1, 10, 3, step=1, key="catboost_l2_leaf_reg")
            thread_count = st.number_input('Número de threads:', min_value=-1, value=-1, key="catboost_thread_count")

            if st.button("Adicionar Modelo à Lista", key="catboost_button"):
                model = CatBoostClassifier(
                    iterations=iterations,
                    random_seed=random_seed,
                    verbose=False,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    thread_count=thread_count
                )
                
                """model = CatBoostClassifier(
                    iterations=100,
                    random_seed=42,
                    verbose=False,
                    learning_rate=0.1,
                    depth=6,
                    l2_leaf_reg=3,
                    thread_count=-1
                )"""
                
                if 'models' not in st.session_state:
                    st.session_state["models"] = {}
                    
                st.session_state["models"].update({"CatBoost": model})
                st.success("Modelo adicionado à lista!")