import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re
import json
from typing import List

def get_document_chunks(file_path, chunk_size=500, chunk_overlap=50):
    """
    Splits a document into chunks of text based on the specified chunk size and overlap.

    This function loads a DOCX file from the given file path, splits its content into smaller chunks 
    using the RecursiveCharacterTextSplitter, and returns a list of text chunks. Each chunk has a 
    maximum size of `chunk_size` characters with an optional overlap between consecutive chunks.

    Args:
        file_path (str): The path to the DOCX file to be loaded and split into chunks.
        chunk_size (int, optional): The maximum number of characters in each chunk (default is 500).
        chunk_overlap (int, optional): The number of overlapping characters between consecutive chunks (default is 50).

    Returns:
        list of str: A list of text chunks extracted from the document.
    """
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def persist_chroma(doc_paths: List[str],
                   persist_path: str):
    """
    Loads, chunks, embeds, and stores documents in a Chroma database with persistence.

    This function processes a list of document file paths, splits each document into chunks, 
    generates embeddings using the OpenAI 'text-embedding-ada-002' model, and stores the 
    chunks and their embeddings in a Chroma database. The database is persisted at the specified 
    `persist_path`. Each document is stored with its metadata, including the title and text content.

    Args:
        doc_paths (List[str]): A list of file paths to the documents (in DOCX format) to be processed.
        persist_path (str): The file path where the Chroma database will be stored.

    Returns:
        None: This function does not return any values, but prints a success message when the process is complete.
    """
    client = chromadb.Client()
    client = chromadb.PersistentClient(path=persist_path)
    
    model = OpenAIEmbeddings(model='text-embedding-ada-002')
    
    collection_name = "sanjay_sarma"
    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        print(f"Error creating or getting collection: {e}")
    
    for doc_path in doc_paths:
        filter_pattern = r"(?<=./data/)(.*?)(?=\.docx)"
        doc_title = re.search(filter_pattern, doc_path).group(0)
        
        chunks = get_document_chunks(doc_path)
        ids = [(doc_title + str(i)) for i in range(len(chunks))]
        metadata = [{'title': doc_title, 'text': chunk} for chunk in chunks]
        embeddings = model.embed_documents(chunks)
        
        collection.add(
            ids = ids,
            documents=chunks,
            metadatas=metadata,
            embeddings=embeddings
        )
    
    print("Documents have been loaded, chunked, embedded, and stored in the database successfully!")
    
def response_filter(letta_response) -> str:
    """
    Extracts the agent reply from a letta_response object.

    Args:
        letta_response (LettaResponse): The response object containing a list of messages.

    Returns:
        str: The extracted reply from a 'send_message' tool call, or a default error message if not found.
    """
    agent_reply = 'There was an internal error, could you please try again?'

    for message in letta_response.messages:
        if message.message_type == 'tool_call_message':
            if message.tool_call.name == 'send_message':
                agent_reply = json.loads(message.tool_call.arguments)['message']
            
    return agent_reply