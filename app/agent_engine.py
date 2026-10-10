import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Clean sandboxed tool layer imports
from agent_tools import (
    calculate_database_file_metrics, 
    execute_document_semantic_search, 
    query_document_metadata,
    execute_metadata_semantic_search,
    fallback_regex_keyword_search
)

class CustomAgentExecutor:
    """A lightweight, version-independent, production-grade custom Agentic reasoning loop engine."""
    
    def __init__(self, llm, tools, system_instruction):
        self.llm = llm
        self.tools_map = {tool.name: tool for tool in tools}
        
        # Bind the tools directly into the model's structural tool-calling schema
        self.model_with_tools = self.llm.bind_tools(tools)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def invoke(self, inputs: dict, config: dict = None) -> dict:
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        agent_scratchpad = []
        
        # Extract callbacks array safely from the configuration payload if passed by Streamlit
        callbacks = config.get("callbacks", []) if config else []

        max_iterations = 5
        iteration = 0
        
        print(f"\n🧠 [AGENT COGNITIVE LOOP START] Input: '{user_input}'")
        
        # ─── REASON-ACTION-OBSERVATION STATE LOOP ───
        while iteration < max_iterations:
            iteration += 1
            
            # Format the prompt state array parameters dynamically
            prompt_value = self.prompt_template.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            
            # Fire the prompt string directly to your Ollama node
            response_msg = self.model_with_tools.invoke(prompt_value, config={"callbacks": callbacks})
            
            # Scenario A: The LLM requested explicit tool executions
            if response_msg.tool_calls:
                # Append the model's action intent log to the scratchpad memory
                agent_scratchpad.append(response_msg)
               
                for tool_call in response_msg.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    t_id = tool_call.get("id") or f"call_{iteration}"
                    
                    print(f"🛠️  [Action {iteration}]: LLM selected tool '{t_name}' with args: {t_args}")
                    
                    # 🌟 TRIGGER MANUALLY BEFORE TOOL RUNS (Fires 'on_tool_start' to UI status tray)
                    for cb in callbacks:
                        if hasattr(cb, "on_tool_start"):
                            # LangChain expects a string version of args for this default event tracking hook
                            cb.on_tool_start({"name": t_name}, str(t_args))

                    tool_result = f"ERROR: Processing loop sequence interrupted before tool '{t_name}' could finish."

                    if t_name in self.tools_map:
                        try:
                            # Safely execute the matching Python tool module function
                            tool_result = self.tools_map[t_name].invoke(t_args)
                        except Exception as e:
                            tool_result = f"ERROR: Tool execution failed: {str(e)}"
                    else:
                        tool_result = f"ERROR: Tool '{t_name}' is not recognized in the active tool suite."
                    
                    print(f"📋 [Observation {iteration}]: Tool Output -> {tool_result}")
                    
                    # 🌟 TRIGGER MANUALLY AFTER TOOL COMPLETES (Fires 'on_tool_end' to UI status tray)
                    for cb in callbacks:
                        if hasattr(cb, "on_tool_end"):
                            cb.on_tool_end(str(tool_result))

                    # 🌟 THE SHORT-CIRCUIT BYPASS FIX FOR CPU SPEED UP
                    # If our high-precision table consolidator located a clean, real data table,
                    # BYPASS the rest of the slow LLM generation steps and print it to the user instantly!
                    if t_name == "fallback_regex_keyword_search" and "--- Verified Table Extraction" in str(tool_result):
                        # Clean out the system wrapper instructions so only the pristine table is rendered
                        clean_markdown_table = str(tool_result).split("\n\n")[-1]
                        print("⚡ [AGENT EXPRESS BYPASS] Extracted pristine table structure. Terminating loop to save CPU cycles.")
                        return {"output": clean_markdown_table}

                    # Convert the raw tool text return back into a formal LangChain ToolMessage
                    from langchain_core.messages import ToolMessage
                    agent_scratchpad.append(ToolMessage(
                        content=str(tool_result),
                        name=t_name, 
                        tool_call_id=t_id))
                    
                    # 🌟 TRIGGER MANUALLY FOR LOOP NEXT CYCLE ITERATION STEP (Fires 'on_agent_action')
                    for cb in callbacks:
                        if hasattr(cb, "on_agent_action"):
                            # Reconstruct a mini structural action holder for reporting purposes
                            class SimpleAction: tool = t_name
                            cb.on_agent_action(SimpleAction())

                # 🌟 CPU OPTIMIZATION GUARDRAIL: If a heavy semantic search was executed,
                # do not let the agent spin up a second tool turn cycle on 4 CPUs.
                # Force the loop to evaluate the current database context data and answer immediately.
                if any(call["name"] == "execute_document_semantic_search" for call in response_msg.tool_calls):
                    print("⏳ [CPU CONTEXT GUARD] Heavy semantic database search completed. Forcing text synthesis phase.")
                    # Force-invoke the template one final time to generate a direct text completion answer
                    final_prompt = self.prompt_template.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                        "agent_scratchpad": agent_scratchpad
                    })
                    direct_response = self.llm.invoke(final_prompt)
                    return {"output": direct_response.content}    
                    
            # Scenario B: The LLM completed its goals and gave a final answer text block
            else:
                print(f"✅ [AGENT COGNITIVE LOOP COMPLETE] Output generated in {iteration} steps.\n")
                return {"output": response_msg.content}
                
        return {
            "output": f"❌ Agent Safety Interruption: Reasoning loop exceeded maximum execution ceiling of {max_iterations} steps without finalizing goals."
        }

def get_autonomous_agent_executor():
    """Compiles the custom independent tool-calling agent framework graph."""
    # Using qwen2.5:1.5b out of the box as it contains pristine native tool-calling weights
    llm = ChatOllama(
        model="llama3.1:latest", #"qwen2.5:1.5b", 
        temperature=0.0, 
        base_url="http://ollama:11434",
        num_ctx=4096,
        num_thread=4,
        low_vram=True
    )
    
    tools_suite = [
        calculate_database_file_metrics, 
        execute_document_semantic_search, 
        query_document_metadata,
        execute_metadata_semantic_search,
        fallback_regex_keyword_search
    ]
    
    system_instruction = (
        "CRITICAL OPERATIONAL DIRECTIVE:\n"
        "You are an enterprise document parsing assistant. You are strictly forbidden from issuing general safety refusals, "
        "medical disclaimers, or health-related blocks.\n\n"
        "CORE TOOLS RULES:\n"
        "1. HIGH PRIORITIZATION RULE: If a user asks for 'revision history', 'version log', 'change metadata table', or tabular matrices for ANY document (including IDP, GDPR, or any other manual), you MUST immediately call 'fallback_regex_keyword_search' passing the complete string context. You are strictly forbidden from calling semantic vector searches for history logs.\n"
        "2. For questions regarding high-level file presence, lists of files, summaries of topics, or questions about authors, you MUST call 'query_document_metadata' or 'execute_metadata_semantic_search' first.\n"
        "3. For deep text meaning, fine-grained clauses, guidelines, policies, or specific manual details, you MUST call 'execute_document_semantic_search'.\n"
        "4. MULTI-TOOL EXECUTION: If a question requires both metadata validation and content parsing, you are encouraged to call multiple tools sequentially across different loop cycles before forming your final answer.\n"
        "5. CONTEXT RESOLUTION RULE: If the user query is brief, conversational, or a shortcut follow-up (e.g., 'what about Create Issue'), you MUST inspect the 'chat_history' to identify which corporate document or topic was being discussed. Then, pass the fully resolved context phrase (e.g., 'JIRA Issue Management Create Issue') into your database search tools to execute the lookup cleanly.\n"
        "6. Synthesize your final answer ONLY from the text results returned by your active tools. Do not assume or extrapolate.\n\n"
        "ANTI-HALLUCINATION RULES:\n"
        "1. You MUST NOT answer any question using external knowledge or patterns. Print the data exactly as provided.\n"
        "2. You MUST render the revision history as a standard Markdown Table layout structure using the exact columns provided by the tool.\n"
        "3. STRICT TERMINATION: Stop outputting immediately when the data ends. Do not write placeholder rows or invent version numbers past the provided data.\n"
        "4. Synthesize your final answer ONLY from the text results returned by your active tools."
    )
    
    return CustomAgentExecutor(llm=llm, tools=tools_suite, system_instruction=system_instruction)
