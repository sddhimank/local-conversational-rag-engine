import os
import streamlit as st
#import datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from agent_engine import get_autonomous_agent_executor
from agent_ingest import process_file_agentically

st.set_page_config(page_title="Agentic AI Troubleshooting Sandbox", page_icon="🤖", layout="wide")

# Establish a clear layout division using a sidebar panel configuration
with st.sidebar:
    st.header("📂 Document Ingestion Panel")
    st.subheader("Autonomous Agent Upload")
    st.caption("Files dropped here are automatically triaged, cleaned, and enriched by llama3.1 before database vector caching.")
    
    # ─── THE DIRECT UI INGESTION SYSTEM ───
    uploaded_file = st.file_uploader(
        "Upload a PDF Manual/Policy", 
        type=["pdf"], 
        accept_multiple_files=False,
        key="agentic_file_uploader"
    )
    
    if uploaded_file is not None:
        # Define the path target matching your agentic storage folder structure
        TARGET_DIR = "./agent_data"
        os.makedirs(TARGET_DIR, exist_ok=True)
        save_path = os.path.join(TARGET_DIR, uploaded_file.name)
        
        # Guardrail: Check if the file has already been written to prevent redundant processing loop cycles
        if not os.path.exists(save_path):
            with st.spinner(f"📥 Writing '{uploaded_file.name}' to cloud storage..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("💾 File written cleanly to VM disk storage!")
                except Exception as e:
                    st.error(f"❌ Storage Write Failure: {str(e)}")
            
            # Instantly pass the local asset straight into your autonomous parsing engine loop
            with st.spinner("🧠 Agent is triaging, chunking, and indexing document parameters..."):
                try:
                    process_file_agentically(save_path)
                    st.success("✅ Ingestion successfully completed! Document is active in vector indices.")
                except Exception as e:
                    st.error(f"❌ Agent Ingestion Failure: {str(e)}")
        else:
            st.info("ℹ️ Document signature already exists in storage registry cache.")

st.title("🤖 Autonomous Agentic AI RAG Sandbox")
st.caption("Discrete parallel playground exploring multi-tool autonomous reasoning loops entirely offline.")

# Instantiate the Agentic core executor once globally via Streamlit resource cache
@st.cache_resource
def load_agent_runtime():
    return get_autonomous_agent_executor()

agent_executor = load_agent_runtime()

# Initialize session messages memory state array if missing
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Streamlit chat interface render loop
for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Issue an autonomous agentic directive..."):
    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert past history blocks into formal LangChain chat message format tracking lists
    history_tracking_list = []
    for m in st.session_state.agent_messages[:-1]:
        if m["role"] == "user":
            history_tracking_list.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history_tracking_list.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("🧠 Agent is thinking, determining task execution strategy..."):
            try:
                # Trigger the autonomous tool reasoning loop
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": history_tracking_list
                })
                final_answer = response.get("output", "Error: Agent completed execution but failed to synthesize text output.")
            except Exception as e:
                final_answer = f"❌ Agent Cognitive Loop Interruption Error: {str(e)}"
        
        st.markdown(final_answer)
        st.session_state.agent_messages.append({"role": "assistant", "content": final_answer})
