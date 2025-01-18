# RAG tool for Letta
def get_sanjay_information(query: str):
    """
    Retrieves relevant information related to a query.

    Args:
        query (str): The query string for which information is being retrieved.

    Returns:
        str: The concatenated content of the relevant search results based on the query.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    
    vdb_embed_model = OpenAIEmbeddings(model='text-embedding-ada-002')
    vectordb = Chroma(
        collection_name='sanjay_sarma',
        embedding_function=vdb_embed_model,
        persist_directory="./chromadb")
    
    text = ""
    results = vectordb.similarity_search(query, k=5)
    for result in results:
        text += result.page_content
    return text