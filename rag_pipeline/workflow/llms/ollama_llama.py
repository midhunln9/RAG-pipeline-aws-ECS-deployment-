from rag_pipeline.workflow.protocols.llm_protocol import LLMProtocol
from langchain_core.messages import BaseMessage
from typing import List
from rag_pipeline.workflow.configs.llm_config import LLMConfig

from langchain_ollama import ChatOllama

class OllamaLLM(LLMProtocol):
    def __init__(self, config: LLMConfig):
        self.model = ChatOllama(model=config.model_name)

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        return self.model.invoke(messages)