import streamlit as st

st.title("Reasoning History")

for reasoning in st.session_state.reasoning_history:
    st.write(reasoning)