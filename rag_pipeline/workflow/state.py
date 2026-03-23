from typing import TypedDict, Optional, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    """
    Represents the state of an agent.
    """
    query: str
    rewritten_query: Optional[str]
    retrieved_documents: Optional[List[Document]]
    session_id: str
    conversation_history: Optional[List[BaseMessage]]
    response: Optional[str]
    summary_before_last_five_messages: Optional[str]