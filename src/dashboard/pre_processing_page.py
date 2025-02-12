import streamlit as st
import logging
import uuid
from pipelines.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.balancing_strategy import DataBalancingStrategy, SmoteBalancing, RandomOversamplingBalancing, RandomUndersamplingBalancing, CombinedSamplingBalancing
from config.inject_logger import inject_logger


import streamlit as st
import logging
import uuid

class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_container, all_logs_container):
        super().__init__()
        self.log_container = log_container  
        self.all_logs_container = all_logs_container  
        self.log_text = ""

    def emit(self, record):
        log_entry = self.format(record)
        self.log_text += log_entry + "\n"
        
        self.log_container.text_area("🔴 Log Atual", "🔴 " + log_entry, height=100, max_chars=10000, key=str(uuid.uuid4()))
        self.all_logs_container.text_area("📝 Logs Acumulados", self.log_text, height=300,  key=str(uuid.uuid4()), disabled=True)

@inject_logger
class PreprocessingPage:
    def __init__(self):
        self.type_database = None
        self.type_strategy = None
        self.collect_new_data = True
        self.teste_size = 0.2
        self.valid_size = 0.2
        self.debug = True
        self.submit_is_true = False

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
                
            col1, col2, col3 = st.columns(3)
            with col1:
                self.teste_size = st.slider("Escolha um valor para o Teste Size", 0.0, 0.5, 0.2, step=0.1)
            with col2:
                self.valid_size = st.slider("Escolha um valor para o Valid Size", 0.0, 0.5, 0.2, step=0.1)
            with col3:
                self.debug = st.checkbox("Definir como Debug")
                
            submitted = st.form_submit_button("Pre-processing")
            
            if submitted and self.valid_test_size_and_valid_size():
                self.submit_is_true = True

        if self.submit_is_true:
            if self.debug:
                    self.process_form_debug()
            else:
                self.process_form()
                
            self.submit_is_true = False

    def process_form(self):
        try:
            with st.spinner("Pré-processando..."):
                log_container = st.empty()
                all_logs_container = st.empty()
                
                streamlit_handler = StreamlitLogHandler(log_container, all_logs_container)
                streamlit_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

                root_logger = logging.getLogger()
                if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
                    root_logger.addHandler(streamlit_handler)

                logger = logging.getLogger(__name__)
                logger.addHandler(streamlit_handler)
                logger.setLevel(logging.INFO)
                
                pipeline = self.define_pipeline()
                X_train, X_valid, X_test, y_train, y_valid, y_test = pipeline.process_data()
                
                st.session_state.X_train = X_train
                st.session_state.X_valid = X_valid
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_valid = y_valid
                st.session_state.y_test = y_test
                st.session_state.processing = True
                
                
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento: {str(e)}")
            st.error(f"❌ Ocorreu um erro durante o pré-processamento. Error: {e}.")
            st.session_state.processing = False
            
            
    
    def process_form_debug(self):
        with st.spinner("Pré-processando..."):
            # Placeholder para os logs
            log_container = st.empty()
            all_logs_container = st.empty()
            
            streamlit_handler = StreamlitLogHandler(log_container, all_logs_container)
            streamlit_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

            root_logger = logging.getLogger()
            if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
                root_logger.addHandler(streamlit_handler)

            logger = logging.getLogger(__name__)
            logger.addHandler(streamlit_handler)
            logger.setLevel(logging.INFO)
            
            pipeline = self.define_pipeline()
            pipeline.process_data()
            
        st.success("✅ Pré-processamento concluído!")
    

    def get_strategy(self) -> DataBalancingStrategy:
        strategies = {
            "Smote": SmoteBalancing(),
            "RandomOverSampler": RandomOversamplingBalancing(),
            "RandomUnderSampler": RandomUndersamplingBalancing(),
            "Combined Sampling": CombinedSamplingBalancing(),
        }
        return strategies.get(self.type_strategy, None)

    def valid_test_size_and_valid_size(self):
        if (self.teste_size + self.valid_size) > 0.9:
            st.error(f"A soma do Teste Size ({self.teste_size}) e Valid Size ({self.valid_size}) é maior que 0.9.")
            return False
        return True
    
    def define_pipeline(self):
        
        self.logger.info("🔄 Iniciando pré-processamento...")
        self.logger.info(f"📌 Dataset: {self.type_database}")
        self.logger.info(f"📌 Estratégia de Balanceamento: {self.type_strategy}")
        self.logger.info(f"📌 Teste Size: {self.teste_size}")
        self.logger.info(f"📌 Valid Size: {self.valid_size}")
        
        return PreprocessingPipeline(
            collect_new_data=self.collect_new_data,
            dataset_type=self.type_database,
            balance_strategy=self.get_strategy(),
            random_state=42,
            test_size=self.teste_size,
            valid_size=self.valid_size
        )
    
    def create_download_button(self, all_logs_container):
        download_key = str(uuid.uuid4())

        current_logs = all_logs_container.text_area("📝 Logs Acumulados", "", height=200, key="all_logs", disabled=True)
        
        all_logs_container.download_button(
            label="💾 Baixar Logs",
            data=current_logs,
            file_name="logs_acumulados.txt",
            mime="text/plain",
            key=download_key  # Passa a chave única aqui
        )
        