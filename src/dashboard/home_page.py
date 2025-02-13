import streamlit as st

class HomePage:
    def render(self):
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