"""
OpenAI LLM implementation.

Provides integration with OpenAI for hosted LLM inference.
"""

import os
import threading

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from rag_pipeline.workflow.configs.llm_config import LLMConfig
from rag_pipeline.workflow.protocols.llm_protocol import LLMProtocol


class OpenAILLM(LLMProtocol):
    """
    LLM adapter for OpenAI hosted inference.

    Wraps LangChain's ChatOpenAI for use with the RAG pipeline.
    """

    _semaphore = threading.Semaphore(10)

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI LLM.

        Args:
            config: LLM configuration containing model name.
        """
        self.model = ChatOpenAI(
            model=config.openai_model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=30,
            max_retries=2,
        )

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with a list of messages.

        Args:
            messages: List of messages to send to the LLM.

        Returns:
            LLM response as a BaseMessage.
        """
        with self._semaphore:
            return self.model.invoke(messages)