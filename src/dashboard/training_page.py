
import streamlit as st
from dashboard.models import CreateModelDecisionTree, CreateModelRandomForestClassifier, CreateModelCatBoost
from pipelines.model_pipeline import ModelingPipeline
from config.inject_logger import inject_logger
import streamlit as st
import logging
import uuid



@inject_logger
class TrainingPage:
    
    def __init__(self):
        if 'models_list' not in st.session_state:
            st.session_state.models_list = []
    
    def render(self):
        
        if "pre_processing" in st.session_state and st.session_state["pre_processing"]:
            st.header("Configuração e Gerenciamento de Modelos")
            st.container()
            st.container()
            CreateModelRandomForestClassifier().create()
            CreateModelDecisionTree().create()
            CreateModelCatBoost().create()

            st.header("Lista de Modelos Adicionados")

            with st.expander("Ver modelos na lista"):
                if "models" in st.session_state and st.session_state.models:
                    for i, (key, value) in enumerate(st.session_state.models.items()):
                        st.write(f"Modelo {i + 1}: {value}")
                        if st.button(f"Remover Modelo {i + 1}"):
                            del st.session_state.models[key]
                            st.rerun()  
                else:
                    st.warning("Nenhum modelo foi adicionado ainda.")
            self.show_process_model()
        else:
            st.warning("Para treinar os modelos é necessário pré-processar antes.")
    
    def show_process_model(self):
        if "models" in st.session_state and bool(st.session_state["models"]):
            with st.form("training_form"):
                st.write("Clique no botão abaixo para iniciar o treinamento.")
                submit = st.form_submit_button("Iniciar processamento")
                if submit:
                    st.session_state["submit_is_valid"] = True
                    
        if "submit_is_valid" in st.session_state and st.session_state["submit_is_valid"]:
            log_container = st.empty()
            all_logs_container = st.empty()
            self.run_model(log_container, all_logs_container)
            st.session_state["submit_is_valid"] = False
            
        
        if "_comparison_df_last" in st.session_state:
            st.dataframe(st.session_state._comparison_df_last)


    def run_model(self, log_container, all_logs_container):
        
        X_train = st.session_state.X_train
        X_valid = st.session_state.X_valid
        X_test  = st.session_state.X_test
        y_train = st.session_state.y_train
        y_valid = st.session_state.y_valid
        y_test  = st.session_state.y_test
        
        meus_modelos = list(st.session_state.models.items())

        try:
            with st.spinner("Treinando o modelo..."):
                self.setup_logging(log_container, all_logs_container)

                pipeline = ModelingPipeline(meus_modelos)
                classes = sorted(y_train.unique())

                _, comparison_df = pipeline.run_pipeline(
                    X_train, X_valid, X_test,
                    y_train, y_valid, y_test,
                    classes=classes
                )
                
                st.session_state["_comparison_df_last"] = comparison_df       
                st.success("Treinamento finalizado!")
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Erro no pré-processamento: {str(e)}")
            st.error(f"❌ Ocorreu um erro durante o pré-processamento. Error: {e}.")
            st.session_state.pre_processing = False
        
    def setup_logging(self, log_container, all_logs_container):
        streamlit_handler = StreamlitLogHandlerTraining(log_container, all_logs_container)
        streamlit_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        root_logger = logging.getLogger()
        root_logger.handlers = []
        
        if not any(isinstance(h, StreamlitLogHandlerTraining) for h in root_logger.handlers):
            root_logger.addHandler(streamlit_handler)
        
        logger = logging.getLogger(__name__)
        logger.addHandler(streamlit_handler)
        logger.setLevel(logging.INFO)
        

class StreamlitLogHandlerTraining(logging.Handler):
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