from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# The state defines what data is passed between nodes in our LangGraph.
# `messages` uses `operator.add` so every new message is appended to the list, not overwritten.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_id: str | None
    escalated: bool
