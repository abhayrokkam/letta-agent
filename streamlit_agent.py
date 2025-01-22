import os
import json
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from modules.engine import create_sanjay_client, get_response
from modules.utils import persist_chroma

# Loading env file
load_dotenv()

# Page config
st.set_page_config(page_title="SanjayBot", page_icon="🧠")
st.title("Sanjay Sarma")

# Creating the VDB if it doesn't exist
persist_path = "./chromadb"
doc_paths = ["./data/Doc1_Sanjay_Info.docx",
             "./data/Doc2_Sanjay_Interview.docx",
             "./data/Doc3_Sanjay_Book.docx"]

if not os.path.isdir(persist_path):
    persist_chroma(persist_path=persist_path,
                   doc_paths=doc_paths)

# Initialize Sanjay client (caching)
@st.cache_resource
def load_client():
    letta_client, agent_state = create_sanjay_client()
    return letta_client, agent_state
letta_client, agent_state = load_client()

# Creating empty chat_history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "reasoning_history" not in st.session_state:
    st.session_state.reasoning_history = []

if "tool_call_history" not in st.session_state:
    st.session_state.tool_call_history = []

# Conversation history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("Sanjay"):
            st.markdown(message.content)

# Conversation
user_query = st.chat_input("Ask Sanjay")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message("User"):
        st.markdown(user_query)
    
    with st.chat_message("Sanjay"):
        client_response = get_response(user_query,
                                        letta_client=letta_client,
                                        agent_state=agent_state)
        
        # Response filtering
        response_messages = json.loads(str(client_response))['messages']
        for message in response_messages:
            if message['message_type'] == 'reasoning_message':
                st.session_state.reasoning_history.append(message['reasoning'])
            if message['message_type'] == 'tool_call_message':
                if message['tool_call']['name'] == 'send_message':
                    agent_reply = json.loads(message['tool_call']['arguments'])['message']
                    st.session_state.chat_history.append(agent_reply)
                else:
                    st.session_state.tool_call_history.append(message['tool_call'])
        
        st.markdown(agent_reply)
    
    st.session_state.chat_history.append(AIMessage(agent_reply))