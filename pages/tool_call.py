import streamlit as st

st.title("Tool Call History")

for tool_call in st.session_state.tool_call_history:
    st.write(tool_call)