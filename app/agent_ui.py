import streamlit as st
#import datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from agent_engine import get_autonomous_agent_executor

st.set_page_config(page_title="Agentic AI Troubleshooting Sandbox", page_icon="🤖", layout="wide")

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
