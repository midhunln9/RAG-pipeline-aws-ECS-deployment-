from typing import Protocol, List
from langchain_core.messages import BaseMessage

class LLMProtocol(Protocol):
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        ...