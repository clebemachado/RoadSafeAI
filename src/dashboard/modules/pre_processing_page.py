import streamlit as st

from ...src.pipeline import PreprocessingPipeline


class PreprocessingPage:
    def __init__(self):
        self.slider_val = None
        self.type_database = None
        self.type_strategy = None
        self.checkbox_val = None

    def render_form(self):
        with st.form("my_form"):
            self.type_database = st.selectbox(
                "Tipo do Dataset",
                ("base", "complete"),
            )
            
            self.type_strategy = st.selectbox(
                "Selecionar Estratégia de Balanceamento",
                ("Smote", "RandomOverSampler", "RandomUnderSampler", "Combined Sampling", "Sem Balanceamento"),
                index=4
            )

            self.checkbox_val = st.checkbox("collect_new_data")

            submitted = st.form_submit_button("Pre-processing")
            if submitted:
                self.process_form()

    def process_form(self):
        """Processa os dados após o envio do formulário."""
        # Aqui você pode manipular os dados ou iniciar a pipeline de pré-processamento
        st.write(f"Slider Value: {self.slider_val}")
        st.write(f"Tipo de Dataset: {self.type_database}")
        st.write(f"Estratégia de Balanceamento: {self.type_strategy}")
        st.write(f"Coletar novos dados: {self.checkbox_val}")
        
        # Exemplo de como integrar com a pipeline (se necessário)
        pipeline = PreprocessingPipeline(
            self.type_database, self.type_strategy, self.slider_val, self.checkbox_val
        )
        pipeline.run()