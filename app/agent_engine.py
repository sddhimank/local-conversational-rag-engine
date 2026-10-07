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
    execute_metadata_semantic_search
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

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        agent_scratchpad = []
        
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
            response_msg = self.model_with_tools.invoke(prompt_value)
            
            # Scenario A: The LLM requested explicit tool executions
            if response_msg.tool_calls:
                # Append the model's action intent log to the scratchpad memory
                agent_scratchpad.append(response_msg)
               
                for tool_call in response_msg.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    t_id = tool_call.get("id") or f"call_{iteration}"
                    
                    print(f"🛠️  [Action {iteration}]: LLM selected tool '{t_name}' with args: {t_args}")
                    
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
                    
                    # Convert the raw tool text return back into a formal LangChain ToolMessage
                    from langchain_core.messages import ToolMessage
                    agent_scratchpad.append(ToolMessage(
                        content=str(tool_result),
                        name=t_name, 
                        tool_call_id=t_id))
                    
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
        base_url="http://ollama:11434"
    )
    
    tools_suite = [
        calculate_database_file_metrics, 
        execute_document_semantic_search, 
        query_document_metadata,
        execute_metadata_semantic_search
    ]
    
    system_instruction = (
        "CRITICAL OPERATIONAL DIRECTIVE:\n"
        "You are an enterprise document parsing assistant. You are strictly forbidden from issuing general safety refusals, "
        "medical disclaimers, or health-related blocks. The term 'sick leave' is a corporate administrative metric, NOT a medical topic.\n\n"
        "CORE RULES:\n"
        "1. You CANNOT answer ANY question using your own general knowledge or personal assumptions.\n"
        "2. You MUST ALWAYS call a tool to inspect the local database index before outputting a response.\n"
        "3. If the user asks about policies, rules, logs, or leaves (e.g., 'sick leaves available'), you MUST immediately "
        "call 'execute_document_semantic_search' passing the exact keywords to pull the corporate manual facts.\n"
        "4. Synthesize your final answer ONLY from the text results returned by your tools. Do not apologize, do not refuse."
    )
    
    return CustomAgentExecutor(llm=llm, tools=tools_suite, system_instruction=system_instruction)
