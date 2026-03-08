from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

from agent.graph import app as graph_app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Setup
app = FastAPI(
    title="Enterprise Triage Agent API",
    description="A LangGraph agent powered by Vertex AI for automated customer support triage.",
    version="1.0.0"
)

from typing import Optional

# Input Models
class ChatRequest(BaseModel):
    user_id: Optional[str] = "Unknown"
    message: str

class ChatResponse(BaseModel):
    response: str
    escalated: bool
    tool_calls_made: list[str]

@app.post("/triage", response_model=ChatResponse)
async def triage_message(request: ChatRequest):
    """
    Main endpoint for the enterprise agent.
    Takes a user message, runs it through the LangGraph cycle, and returns the final LLM response.
    """
    logger.info(f"Received message from {request.user_id}: {request.message}")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "customer_id": request.user_id,
        "escalated": False
    }

    try:
        # 1. Invoke the Agent Graph
        # This will loop LLM -> Tools -> LLM internally until reaching END
        final_state = graph_app.invoke(initial_state)
        
        # 2. Extract the final response
        last_message = final_state["messages"][-1]
        
        # 3. Analytics: See what tools were called during this interaction
        tools_called = []
        for msg in final_state["messages"]:
            if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"):
                for call in getattr(msg, "tool_calls"):
                    tools_called.append(call["name"])

        return ChatResponse(
            response=last_message.content,
            escalated=final_state.get("escalated", False),
            tool_calls_made=tools_called
        )

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail="Agent encounters an internal error.")

@app.get("/health")
def health_check():
    """GCP Cloud Run health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")
