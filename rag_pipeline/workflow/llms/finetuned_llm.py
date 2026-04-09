"""
Finetuned LLM implementation.

Provides integration with a locally stored finetuned Hugging Face LLM.
"""

import threading

from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag_pipeline.workflow.configs.llm_config import LLMConfig
from rag_pipeline.workflow.protocols.llm_protocol import LLMProtocol


class FinetunedLLM(LLMProtocol):
    """
    LLM adapter for local finetuned inference.

    Wraps LangChain's ChatHuggingFace for use with the RAG pipeline.
    """

    _semaphore = threading.Semaphore(10)

    def __init__(self, config: LLMConfig):
        """
        Initialize finetuned LLM.

        Args:
            config: LLM configuration containing local model path.
        """
        model_path = config.finetuned_model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            # remove device_map if you do not want to require accelerate
            # device_map="auto",
        )

        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        self.model = ChatHuggingFace(llm=llm)

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