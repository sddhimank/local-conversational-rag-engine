import os
from datetime import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent_engine import get_autonomous_agent_executor
from agent_ingest import process_file_agentically
from langchain_core.callbacks import BaseCallbackHandler

class StreamlitAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to stream live background agent operations 
    directly into a Streamlit status container layout.
    """
    def __init__(self, status_container):
        self.status = status_container

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.status.write("🧠 **Llama 3.1** is analyzing prompt constraints and planning next steps...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown Tool")
        self.status.write(f"🛠️ **Executing Tool:** `{tool_name}`...")
        self.status.write(f"📥 *Passing search arguments:* `{input_str}`")

    def on_tool_end(self, output, **kwargs):
        self.status.write("📋 **Data Retrieval Complete.** Parsing layout structure for table grids...")
        if "|" in str(output):
            self.status.write("✨ *Detected formatted markdown matrix rows inside the database chunk!*")

    def on_agent_action(self, action, **kwargs):
        self.status.write(f"🔄 **Agent Action Loop:** Decided to call `{action.tool}` based on reasoning matrix constraints.")

st.set_page_config(page_title="Agentic AI Troubleshooting Sandbox", page_icon="🤖", layout="wide")

# Establish a clear layout division using a sidebar panel configuration
with st.sidebar:
    st.header("📂 Document Ingestion Panel")
    st.subheader("Autonomous Agent Upload")
    st.caption("Files dropped here are automatically triaged, cleaned, and enriched by llama3.1 before database vector caching.")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF Manual/Policy", 
        type=["pdf"], 
        accept_multiple_files=False,
        key="agentic_file_uploader"
    )
    
    if uploaded_file is not None:
        TARGET_DIR = "./agent_data"
        os.makedirs(TARGET_DIR, exist_ok=True)
        save_path = os.path.join(TARGET_DIR, uploaded_file.name)
        
        if not os.path.exists(save_path):
            with st.spinner(f"📥 Writing '{uploaded_file.name}' to cloud storage..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("💾 File written cleanly to VM disk storage!")
                except Exception as e:
                    st.error(f"❌ Storage Write Failure: {str(e)}")
            
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

@st.cache_resource
def load_agent_runtime():
    return get_autonomous_agent_executor()

agent_executor = load_agent_runtime()

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Streamlit chat interface render loop (renders historical turns instantly)
for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── ACTION TRIGGER BRANCH ───
if user_input := st.chat_input("Issue an autonomous agentic directive..."):
    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert past history blocks into formal LangChain chat message objects
    history_tracking_list = []
    for m in st.session_state.agent_messages[:-1]:
        if m["role"] == "user":
            history_tracking_list.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history_tracking_list.append(AIMessage(content=m["content"]))

    # ✅ FIXED SCOPE: Assistant reasoning loop now executes strictly when a prompt arrives
    with st.chat_message("assistant"):
        response_container = st.container()
        
        with st.status("🚀 Agent Processing Pipeline Live Track...", expanded=True) as status_box:
            ui_callback = StreamlitAgentCallbackHandler(status_box)
            
            try:
                start_time = datetime.now()
                
                response = agent_executor.invoke(
                    {
                        "input": user_input,
                        "chat_history": history_tracking_list
                    },
                    config={"callbacks": [ui_callback]}
                )
                
                end_time = datetime.now()
                elapsed_time = (end_time - start_time).total_seconds()
                
                final_answer = response.get("output", "")
                status_box.update(label=f"✅ Reasoning Loop Complete ({elapsed_time:.2f}s)", state="complete", expanded=False)
                
            except Exception as e:
                final_answer = f"❌ Agent Pipeline Interruption: {str(e)}"
                elapsed_time = 0.0
                status_box.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)

        with response_container:
            st.markdown(final_answer)
            if elapsed_time > 0.0:
                st.caption(f"⏱️ **Total Turn Latency:** `{elapsed_time:.2f} seconds`")

        st.session_state.agent_messages.append({"role": "assistant", "content": final_answer})
