import os
import json
from dotenv import load_dotenv

load_dotenv()
import requests
import google.auth
import google.auth.transport.requests
from typing import Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI, VertexAIModelGarden
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import LLM
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from agent.state import AgentState
from agent.tools import tools

# Custom wrapper to handle Model Garden deployment
class GemmaModelGardenLLM(BaseChatModel):
    endpoint_url: str

    def __init__(self, endpoint_url: str, **kwargs: Any):
        super().__init__(endpoint_url=endpoint_url, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "gemma_model_garden"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Extract the last user message
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        # Extract tool results (RAG context)
        tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
        raw_context = "\n".join(tool_results) if tool_results else ""

        # Smart Context Filtering: isolate paragraphs with keywords ONLY if context is large
        context = ""
        if raw_context:
            if len(raw_context) < 3000:
                context = raw_context
            else:
                query_words = last_user_msg.lower().split()
                relevant_paragraphs = []
                paragraphs = raw_context.split("\n")
                
                for i, p in enumerate(paragraphs):
                    if any(word in p.lower() for word in query_words if len(word) > 2):
                        # Include current and next line for context
                        relevant_paragraphs.append(p.strip())
                        if i + 1 < len(paragraphs):
                            relevant_paragraphs.append(paragraphs[i+1].strip())
                
                context = "\n".join(relevant_paragraphs) if relevant_paragraphs else raw_context[:2000]

        if context:
            # Stable RAG Prompt
            prompt = f"Facts:\n{context}\n\nQuestion: {last_user_msg}\n\nAnswer based ONLY on the Facts above. If not mentioned, say I don't know.\n\nAnswer: {last_user_msg} is"
        else:
            # FIRST TURN: Include instructions to FORCE tool calling
            prompt = f"""<start_of_turn>user
You are a Support Bot.
Rules:
1. ALWAYS use 'search' for questions about PEOPLE or POLICIES.
2. Format: Tool: search(query="...")

Question: {last_user_msg}<end_of_turn>
<start_of_turn>model
"""
        
        creds, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)

        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "max_output_tokens": 1024,
                "temperature": 0.2, # Slightly more creative might break the truncation cycle
                "top_p": 0.95,
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds.token}"
        }

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            
            resp_json = response.json()
            raw_prediction = resp_json["predictions"][0]
            
            content = raw_prediction
            # Handle continuation prompt echo
            if "Answer:" in content:
                content = content.split("Answer:")[-1].strip()
            
            # Remove "Question is" echo if hallucinated by model
            if " is" in content[:50] and last_user_msg.lower() in content.lower()[:50]:
                content = content.split(" is", 1)[-1].strip()

            if "<start_of_turn>model\n" in content:
                content = content.split("<start_of_turn>model\n")[-1].strip()
            
            # Final cleaning
            content = content.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
            if content.startswith("Output:"):
                content = content[7:].strip()
            if content.startswith("Answer:"):
                content = content[7:].strip()
                
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])
        except Exception as e:
            print(f"Error calling Vertex AI: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        return self

# 1. Initialize the LLM
endpoint_id = os.getenv("VERTEX_MODEL_ENDPOINT_ID")
project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "us-central1")
dedicated_endpoint = os.getenv("VERTEX_DEDICATED_ENDPOINT")

if not endpoint_id or not dedicated_endpoint:
    raise ValueError("VERTEX_MODEL_ENDPOINT_ID and VERTEX_DEDICATED_ENDPOINT must be set in .env")

# Use verified Dedicated DNS for custom Model Garden deployment
endpoint_url = f"https://{dedicated_endpoint}/v1/projects/{project_id}/locations/europe-west4/endpoints/{endpoint_id}:predict"
llm = GemmaModelGardenLLM(endpoint_url=endpoint_url)
print(f"Connected to Vertex Model: {endpoint_id}")

# Bind our custom tools to the LLM so it knows it can call them
llm_with_tools = llm.bind_tools(tools)

# 2. Define the Nodes
def agent_node(state: AgentState):
    """The brain of the agent. It looks at the message history and decides whether to respond directly or call a tool."""
    messages = state["messages"]
    
    # Prepend a system prompt if it's the first message
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content="""You are a Support Bot.
Rules:
1. ALWAYS use 'search' for questions about PEOPLE or POLICIES.
2. If 'CONTEXT' is provided, you MUST use it to answer.
3. Your answer must be detailed and based ONLY on the context.
- Format: Tool: search(query="...")
""")
        messages = [sys_msg] + messages
    
    # Call the LLM
    print(f"\n--- INVOKING GEMMA ---")
    response = llm_with_tools.invoke(messages)
    print(f"--- GEMMA RESPONSE: {response.content} ---")
    
    # Manual tool call parsing for models that don't support native tool calling (like Gemma 2B)
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        content = response.content
        if any(x in content.lower() for x in ["tool:", "search(", "lookup(", "ticket("]):
            import re
            # Extract Tool: name, handling optional markdown bolding and case-insensitivity
            match = re.search(r"(?:\*\*|)?Tool:(?:\*\*|)?\s*(\w+)", content, re.IGNORECASE)
            # Or if it just wrote the function call
            if not match:
                match = re.search(r"(\w+)\(", content)
            
            if match:
                partial_name = match.group(1).lower()
                
                # Mapping short names or partials to actual tools
                tool_mapping = {
                    "search": "search_knowledge_base",
                    "lookup": "lookup_customer",
                    "ticket": "create_escalation_ticket"
                }
                
                tool_name = None
                for k, v in tool_mapping.items():
                    if partial_name.startswith(k):
                        tool_name = v
                        break
                
                if tool_name:
                    args = {}
                    # Try to extract args from content
                    arg_match = re.search(r'query\s*=\s*"([^"]*)', content)
                    if arg_match:
                        args["query"] = arg_match.group(1)
                    
                    # FALLBACK: If query is required but missing, or Tool: was provided without args
                    if tool_name == "search_knowledge_base" and not args.get("query"):
                        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
                        if user_messages:
                            args["query"] = user_messages[-1].content
                    
                    # Similar fallbacks for other tools if needed
                    if tool_name == "lookup_customer" and not args.get("customer_id"):
                        import re
                        customer_id_match = re.search(r"CUST-\d+", str(messages))
                        if customer_id_match:
                            args["customer_id"] = customer_id_match.group(0)

                    # Assign tool_calls to the response object
                    response.tool_calls = [
                        {"name": tool_name, "args": args, "id": f"call_{hash(content) % 1000}"}
                    ]
        
        # FINAL SAFETY: If it's a first turn and NO tool call was detected, but it's a search-likely query
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            user_msg = [m for m in messages if isinstance(m, HumanMessage)][-1].content.lower()
            search_keywords = ["who is", "what is", "refund", "policy", "sanket", "gde", "google developer expert"]
            
            if "?" in user_msg or any(k in user_msg for k in search_keywords):
                # Only auto-search if we haven't already performed a search in this thread
                if len(messages) <= 3:
                    print(f"[Auto-Search] Forcing search for query: {user_msg}")
                    response.tool_calls = [
                        {"name": "search_knowledge_base", "args": {"query": user_msg}, "id": "auto_search"}
                    ]
    
    # Check if a ticket was created
    escalated = state.get("escalated", False)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            if call["name"] == "create_escalation_ticket":
                escalated = True
                
    return {"messages": [response], "escalated": escalated}

# Langgraph pre-built node to automatically execute the tools the LLM requests
tool_node = ToolNode(tools)

# 3. Define Routing Logic
def should_continue(state: AgentState) -> str:
    """Decide if we go to back to the user or execute a tool."""
    last_message = state["messages"][-1]
    
    # If the LLM requested an action, route to the tools
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, it answered the user directly, stop here.
    return END

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add our two nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Define the flow
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
# After a tool executes, ALWAYS loop back to the agent so it can read the tool output
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable application
app = workflow.compile()
