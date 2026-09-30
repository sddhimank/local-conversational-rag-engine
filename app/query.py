import sys
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
COLLECTION_NAME = "folder_documents"

def format_docs(docs):
    """Formats retrieved context snippets cleanly."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_search(query_str: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    
    # 1. OPTIMIZATION: Switched to llama3.2, added a generous 5-minute timeout window
    # and explicitly lowered context size (num_ctx) to speed up CPU inference.
    llm = ChatOllama(
        model="qwen2.5:1.5b", 
        temperature=0.1, 
        base_url="http://localhost:11434",
        timeout=300.0,
        num_ctx=2048
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
    )
    
    # 2. OPTIMIZATION: Reduced 'k' from 4 to 2 chunks.
    # Less input text means your laptop CPU generates the final answer significantly faster.
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    rag_prompt_template = """Use the provided context fragments to answer the user question.

Context:
{context}

Question: 
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    '''
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    '''
    llm_chain = prompt | llm | StrOutputParser()
    
    print(f"\n🔍 Executing Search for Query: '{query_str}'")
    print("🧠 Fetching context and generating streaming response (Please wait a moment for CPU to compute initial tokens)... \n")
     # Step 1: Manually invoke the retriever layer first to isolate and preserve metadata
    retrieved_docs = retriever.invoke(query_str)
    
    print("💡 Response: ", end="", flush=True)

    # Step 2: Stream the LLM response to the terminal
    # We pass the pre-fetched documents into the chain directly
    context_content = format_docs(retrieved_docs)
    
    # 3. OPTIMIZATION: Switched from .invoke() to .stream()
    # This prints words to your terminal the absolute millisecond they are generated.
    try:
        # FIX: Bypass duplicate DB embedding runs by directly executing prompt downstream steps
        #llm_chain = prompt | llm | StrOutputParser()

        # Stream output mapping the precise expected keys safely
        for chunk in llm_chain.stream({"context": context_content, "question": query_str}):
            print(chunk, end="", flush=True)
        print("\n")

         # Step 3: Parse and print the precise source document attributes cleanly
        print("📋 Sources Referenced:")
        print("-" * 50)

        seen_sources = set()
        for idx, doc in enumerate(retrieved_docs, start=1):
            # Extract metadata dictionaries provided by DirectoryLoader & PyPDFLoader
            metadata = doc.metadata
        
            # Extract pure file name from path string
            source_path = metadata.get("source", "Unknown Document")
            file_name = os.path.basename(source_path)
            
            # LangChain's internal page index starts at index 0, so increment by 1 for readable layout
            page_number = metadata.get("page", None)
            page_str = f"Page {page_number + 1}" if page_number is not None else "Text Snippet/Line"
            
            # Extract a brief preview snippet to help verify document alignment
            text_preview = doc.page_content.strip().replace('\n', ' ')[:90] + "..."
            
            # Construct a unique identifier string to prevent printing identical duplicate pages twice
            source_identifier = f"{file_name} | {page_str}"
            if source_identifier not in seen_sources:
                print(f" [{idx}] 📄 Document: {file_name}")
                print(f"     📍 Location: {page_str}")
                print(f"     ✂️  Excerpt: \"{text_preview}\"\n")
                seen_sources.add(source_identifier)

    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        print("💡 Tip: Ensure 'docker exec -it rag_ollama ollama pull llama3.2' was run completely.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query string.")
    else:
        user_query = " ".join(sys.argv[1:])
        run_rag_search(user_query)
