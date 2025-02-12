import streamlit as st

from pipelines.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.balancing_strategy import DataBalancingStrategy, SmoteBalancing, RandomOversamplingBalancing, RandomUndersamplingBalancing, CombinedSamplingBalancing

class PreprocessingPage:
    def __init__(self):
        self.slider_val = None
        self.type_database = None
        self.type_strategy = None
        self.collect_new_data = None
        self.teste_size = 0.2
        self.valid_size = 0.2

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

            self.collect_new_data = st.checkbox("collect_new_data")
            
            col1, col2 = st.columns(2)

            # Colocar o slider na primeira coluna
            with col1:
                self.teste_size = st.slider('Escolha um valor para o teste size', 0.0, 0.5, 0.2, step=0.1)

            # Colocar algo na segunda coluna
            with col2:
                self.valid_size = st.slider('Escolha um valor para o valid size', 0.0, 0.5, 0.2, step=0.1)


            submitted = st.form_submit_button("Pre-processing")
            
            
            if submitted and  self.get_strategy:
                self.process_form()
                #self.process_form()

    def process_form(self):
        st.write(f"Slider Value: {self.slider_val}")
        st.write(f"Tipo de Dataset: {self.type_database}")
        st.write(f"Estratégia de Balanceamento: {self.type_strategy}")
        st.write(f"Coletar novos dados: {self.checkbox_val}")
        
        pipeline = PreprocessingPipeline(
            collect_new_data=self.collect_new_data,
            dataset_type=self.type_database, 
            balance_strategy=self.get_strategy(), 
            random_state=42,
            test_size = self.teste_size,
            valid_size=self.valid_size
        )
        
        #pipeline.run()
        
    @property
    def get_strategy(self) -> DataBalancingStrategy:
        if self.type_strategy == "Smote":
            return SmoteBalancing()
        elif self.type_strategy == "RandomOverSampler":
            return RandomOversamplingBalancing()
        elif self.type_strategy == "RandomUnderSampler":
            return RandomUndersamplingBalancing()
        elif self.type_strategy == "Combined Sampling":
            return CombinedSamplingBalancing()
        else:
            return None 
    
    
    def valid_test_size_and_valid_size(self):
        if (self.teste_size + self.valid_size) > 0.9:
            st.error(f"A soma do teste size : {self.teste_size} e self.valid_size {self.valid_size} é maior que 0.9")
            return False
        return True