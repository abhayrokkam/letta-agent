import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.title("Chat History with Reasoning")

for message in st.session_state.chat_reasoning_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("Sanjay"):
            st.write(message.content)