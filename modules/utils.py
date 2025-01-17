import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re
from typing import List

def get_document_chunks(file_path, chunk_size=500, chunk_overlap=50):
    """
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