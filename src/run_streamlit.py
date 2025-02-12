import streamlit as st
from streamlit_option_menu import option_menu
from dashboard.pre_processing_page import PreprocessingPage


preProcessingPage: PreprocessingPage = PreprocessingPage() 

def main():
    menuOptions = option_menu(
        "RoadMapAI", ["Exploration", "Preparation", "Training", 'Reports'], 
        menu_icon="cast", default_index=0, orientation="horizontal"
    )
    
    if menuOptions == "Exploration":
        #Fazer parte da visualização de dados
        pass
    
    if menuOptions == "Preparation":
        preProcessingPage.render_form()
    
if __name__ == "__main__":
    main()