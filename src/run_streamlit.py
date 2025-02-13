import streamlit as st
from streamlit_option_menu import option_menu
from dashboard import PreprocessingPage, HomePage, TrainingPage, ReportPage

homePage: HomePage = HomePage()
preProcessingPage: PreprocessingPage = PreprocessingPage() 
trainingPage: TrainingPage = TrainingPage()
reportPage: ReportPage = ReportPage()

def main():
    
    menuOptions = option_menu(
        "RoadMapAI", ["EDA", "Preparation", "Training", 'Reports',], 
         default_index=0, orientation="horizontal"
    )
    
    if menuOptions == "Exploration":
        homePage.render()
        
    if menuOptions == "Preparation":
        preProcessingPage.render()
    
    if menuOptions == "Training":
        trainingPage.render()
    
    if menuOptions == "Reports":
        reportPage.render()
    
if __name__ == "__main__":
    main()