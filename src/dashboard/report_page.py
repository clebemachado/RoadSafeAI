from pathlib import Path
import streamlit as st
import pandas as pd
import json
from PIL import Image

class ReportPage:
    def render(self):
        # Caminho da pasta 'resultados'
        results_path = Path("./resultados")  # Ajuste conforme necessário
        
        subfolders = [f.name for f in results_path.iterdir() if f.is_dir() and f.name != "non_tree_models"]

        if subfolders:
            selected_folder = st.selectbox("Selecione uma execução:", subfolders)
            selected_path = results_path / selected_folder

            files = list(selected_path.iterdir())

            if files:
                for file in files:
                    if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        st.image(Image.open(file), caption=file.name, use_container_width=True)
                    elif file.suffix.lower() in [".csv", ".xlsx"]:
                        df = pd.read_csv(file) if file.suffix.lower() == ".csv" else pd.read_excel(file)
                        st.dataframe(df, use_container_width=True)
            else:
                st.warning("Nenhum arquivo encontrado nesta pasta.")

            # Passa a pasta selecionada como base para o método show_for_method
            self.show_for_method(selected_path)
        else:
            st.error("Nenhuma pasta encontrada em 'resultados'.")

    def show_for_method(self, selected_path):
        """Exibe os métodos dentro da pasta selecionada"""
        methods = [f.name for f in selected_path.iterdir() if f.is_dir()]
        
        if methods:
            selected_method = st.selectbox("Escolha um método:", methods)
            selected_method_path = selected_path / selected_method
            files = list(selected_method_path.iterdir())
            images = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

            if images:
                selected_image = st.selectbox("Escolha uma imagem para visualizar:", [img.name for img in images])
                image_path = selected_method_path / selected_image
                st.image(Image.open(image_path), caption=selected_image, use_container_width=True)

            for file in files:
                if file.suffix.lower() in [".csv", ".xlsx"]:
                    df = pd.read_csv(file) if file.suffix.lower() == ".csv" else pd.read_excel(file)
                    st.write(f"📄 {file.name}")
                    st.dataframe(df)
                elif file.suffix.lower() == ".json":
                    with open(file, "r") as f:
                        json_data = json.load(f)
                    st.write(f"📊 {file.name}")
                    st.json(json_data)
                elif file.suffix.lower() == ".txt":
                    with open(file, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    st.write(f"📜 {file.name}")
                    st.text_area("Conteúdo do arquivo:", text_content, height=200)
        else:
            st.warning("Nenhum método encontrado nesta pasta.")