import streamlit as st
import logging
import uuid
from pipelines.preprocessing_pipeline import PreprocessingPipeline
from config.inject_logger import inject_logger


class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_container, all_logs_container):
        super().__init__()
        self.log_container = log_container
        self.all_logs_container = all_logs_container
        self.log_text = ""

    def emit(self, record):
        log_entry = self.format(record)
        self.log_text += log_entry + "\n"

        self.log_container.text_area(
            "🔴 Log Atual", "🔴 " + log_entry, height=100, max_chars=10000, key=str(uuid.uuid4())
        )
        self.all_logs_container.text_area(
            "📝 Logs Acumulados", self.log_text, height=300, key=str(uuid.uuid4()), disabled=True
        )
        
        st.session_state["log_container_preprocessing"] = self.log_container
        st.session_state["all_logs_container_preprocessing"] = st.empty()


@inject_logger
class PreprocessingPage:
    def __init__(self):
        self.init_parameters()

    def init_parameters(self):
        self.type_database = None
        self.type_strategy = None
        self.collect_new_data = True
        self.teste_size = 0.2
        self.valid_size = 0.2
        self.submit_is_true = False
        self.view_result = False

    def render(self):
        with st.form("my_form"):
            self.setup_form_inputs()
            
            valueButtonStr = "Reprocessar" if "pre_processing" in st.session_state and st.session_state["pre_processing"] else "Pré-Processar"
            submitted = st.form_submit_button(valueButtonStr)
            
            if submitted and self.valid_test_size_and_valid_size() and not self.submit_is_true:
                self.submit_is_true = True

        if self.submit_is_true:
            self.execute_processing()
            
        if "pre_processing" in st.session_state:
            self.results_widgets()

    def setup_form_inputs(self):
        self.type_database = st.selectbox("Tipo do Dataset", ("base", "complete"))
        self.type_strategy = st.selectbox(
            "Selecionar Estratégia de Balanceamento",
            ("Smote", "RandomOverSampler", "RandomUnderSampler", "Combined Sampling", "Sem Balanceamento"),
            index=4,
        )
        col1, col2, _ = st.columns(3)
        with col1:
            self.teste_size = st.slider("Escolha um valor para o Teste Size", 0.0, 0.5, 0.2, step=0.1)
        with col2:
            self.valid_size = st.slider("Escolha um valor para o Valid Size", 0.0, 0.5, 0.2, step=0.1)

    def execute_processing(self):
        self.process_form()
        self.submit_is_true = False

    def process_form(self):
        try:
            with st.spinner("Pré-processando..."):
                log_container, all_logs_container = st.empty(), st.empty()
                
                st.session_state["log_container_preprocessing"] = log_container
                st.session_state["all_logs_container_preprocessing"] = all_logs_container
                
                self.setup_logging(log_container, all_logs_container)
                
                pipeline = self.define_pipeline()
                X_train, X_valid, X_test, y_train, y_valid, y_test = pipeline.process_data()

                self.store_results(X_train, X_valid, X_test, y_train, y_valid, y_test)
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Erro no pré-processamento: {str(e)}")
            st.error(f"❌ Ocorreu um erro durante o pré-processamento. Error: {e}.")
            st.session_state.pre_processing = False

    def setup_logging(self, log_container, all_logs_container):
        streamlit_handler = StreamlitLogHandler(log_container, all_logs_container)
        streamlit_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        root_logger = logging.getLogger()
        root_logger.handlers = []
        
        if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(streamlit_handler)
        
        logger = logging.getLogger(__name__)
        logger.addHandler(streamlit_handler)
        logger.setLevel(logging.INFO)

    def store_results(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
        st.session_state.update(
            {
                "X_train": X_train,
                "X_valid": X_valid,
                "X_test": X_test,
                "y_train": y_train,
                "y_valid": y_valid,
                "y_test": y_test,
                "pre_processing": True,
            }
        )

    def get_strategy(self) -> str:
        
        strategies = {
            "Smote": 'smote',
            "RandomOverSampler": 'random_over',
            "RandomUnderSampler": 'random_under',
            "Combined Sampling": 'combined',
        }
        
        st.warning(f"ESTRATÉGIA: {self.type_strategy}")
        return strategies.get(self.type_strategy, None)

    def valid_test_size_and_valid_size(self):
        if (self.teste_size + self.valid_size) > 0.9:
            st.error(f"A soma do Teste Size ({self.teste_size}) e Valid Size ({self.valid_size}) é maior que 0.9.")
            return False
        return True

    def define_pipeline(self):
        logging.info("🔄 Iniciando pré-processamento...")
        logging.info(f"📌 Dataset: {self.type_database}")
        logging.info(f"📌 Estratégia de Balanceamento: {self.type_strategy}")
        logging.info(f"📌 Teste Size: {self.teste_size}")
        logging.info(f"📌 Valid Size: {self.valid_size}")
        
        return PreprocessingPipeline(
            collect_new_data=self.collect_new_data,
            dataset_type=self.type_database,
            balance_strategy=self.get_strategy(),
            random_state=42,
            test_size=self.teste_size,
            valid_size=self.valid_size,
        )
        
    def results_widgets(self):
        if "pre_processing" in st.session_state:
            if "view_result" not in st.session_state:
                st.session_state.view_result = False  # Inicializa no estado
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Ver resultados", use_container_width=True):
                    st.session_state.view_result = True
            with col2:
                if st.button("Ocultar resultados", use_container_width=True):
                    st.session_state.view_result = False
            with col3:
                if st.button("Limpar dados"):
                    st.session_state.view_result = False
                    self.clean_store()
            
            if "view_result" in st.session_state and st.session_state.view_result:
                self.view_result_widget()
            else:
                st.empty()

    def view_result_widget(self):
        st.subheader("📌 Parâmetros Utilizados")
        st.write(f"**📌 Dataset: {self.type_database}**")
        st.write(f"**📌 Estratégia de Balanceamento: {self.type_strategy}**")
        st.write(f"**📌 Teste Size: {self.teste_size}**")
        st.write(f"**📌 Valid Size: {self.valid_size}**")
        
        st.subheader("📊 Resultados Obtidos")
        st.subheader("📊 Resultados Obtidos")

        if "X_train" in st.session_state:
            st.write("**X_train:**")
            st.write(st.session_state.X_train)

        if "X_valid" in st.session_state:
            st.write("**X_valid:**")
            st.write(st.session_state.X_valid)

        if "X_test" in st.session_state:
            st.write("**X_test:**")
            st.write(st.session_state.X_test)
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "y_train" in st.session_state:
                st.write("**y_train:**")
                st.write(st.session_state.y_train)
                
        with col2:
            if "y_valid" in st.session_state:
                st.write("**y_valid:**")
                st.write(st.session_state.y_valid)
                
        with col3:
            if "y_test" in st.session_state:
                st.write("**y_test:**")
                st.write(st.session_state.y_test)

    def clean_store(self):
        st.session_state.clear()