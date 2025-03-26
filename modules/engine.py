from dotenv import load_dotenv
load_dotenv()

import letta
import letta.client
from letta.schemas.memory import ChatMemory

from .prompts import sanjay_persona, system_prompt
from .tools import get_sanjay_information

def create_sanjay_client():
    """
    Creates a Sanjay agent client using the Letta framework.

    This function sets up a Letta client, configures the default LLM (GPT-4o-mini) and embedding model 
    (text-embedding-ada-002), and creates a tool for retrieving information. It also sets up an agent 
    named 'sanjay_sarma' with a custom memory and updates the system prompt for the agent.

    The function returns the created Letta client and the agent state.

    Returns:
        letta_client (LettaClient): The configured Letta client instance.
        agent_state (AgentState): The state of the created Sanjay agent.
    """
    # Creating the letta_client and setting default LLM and Embedding Model
    letta_client = letta.create_client()
    letta_client.set_default_llm_config(letta.LLMConfig.default_config("o1-mini")) 
    letta_client.set_default_embedding_config(letta.EmbeddingConfig.default_config("text-embedding-ada-002"))

    # Setting agent name
    agent_name = 'sanjay_sarma'

    # Creating a tool
    get_info_tool = letta_client.create_tool(get_sanjay_information)

    # Create sanjay agent
    agent_state = letta_client.create_agent(
        name=agent_name,
        tool_ids=[get_info_tool.id],
        memory=ChatMemory(
            human="",
            persona=sanjay_persona
        )
    )

    # Updating the system prompt
    agent_state.system = system_prompt
    
    print(f"Created agent with name {agent_state.name} and unique ID {agent_state.id}") 
    return letta_client, agent_state

def get_response(query: str,
                 letta_client,
                 agent_state):
    """
    Sends a query message to a client using the provided LettA client and agent state, and retrieves the response.

    Args:
        query (str): The query message to be sent to the client.
        letta_client: The LettA client instance responsible for sending and receiving messages.
        agent_state: The state of the agent, used to identify the agent and manage interactions.

    Returns:
        tuple: A tuple containing:
            - client_response: The full response object from the LettA client after sending the query.
            - reply_message (str): The specific reply message extracted from the client response.
    """
    
    client_response = letta_client.send_message(
        agent_id=agent_state.id, 
        message=query, 
        role="user" 
    )
    
    return client_response