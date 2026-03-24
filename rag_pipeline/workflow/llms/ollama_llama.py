"""
Ollama LLM implementation.

Provides integration with Ollama for local LLM inference.
"""

from typing import List

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from rag_pipeline.workflow.configs.llm_config import LLMConfig


class OllamaLLM:
    """
    LLM adapter for Ollama local inference.
    
    Wraps LangChain's ChatOllama for use with the RAG pipeline.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama LLM.
        
        Args:
            config: LLM configuration containing model name.
        """
        self.model = ChatOllama(model=config.model_name)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with a list of messages.
        
        Args:
            messages: List of messages to send to the LLM.
            
        Returns:
            LLM response as a BaseMessage.
        """
        return self.model.invoke(messages)