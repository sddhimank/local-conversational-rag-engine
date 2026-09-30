import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
COLLECTION_NAME = "folder_documents"

def format_docs(docs):
    """Formats retrieved context snippets cleanly for injection into prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_search(query_str: str):
    # Initialize open-source components matching Ingestion settings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    llm = ChatOllama(model="llama3.2", temperature=0.1, base_url="http://localhost:11434")

    # Connect to the exact vector index table collection
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
    )
    
    # Configure retriever parameters (fetch top 4 matching pieces)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Strict grounding Prompt engineering to stop LLM hallucinations
    rag_prompt_template = """You are an advanced AI assistant powered by a local RAG framework. 
Use strictly the provided context snippets to answer the user request. 
If you do not know the answer based on the context, say explicitly that the context does not contain the answer.

Context:
{context}

Question: 
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    # Establish the unified LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n🔍 Executing Search for Query: '{query_str}'\n")
    response = rag_chain.invoke(query_str)
    print("💡 Response:")
    print(response)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query string. Example: python query.py 'What is the document about?'")
    else:
        user_query = " ".join(sys.argv[1:])
        run_rag_search(user_query)
