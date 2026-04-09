"""LLM configuration."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Language model configuration."""

    model_name: str = "llama3.2" #for ollama
    openai_model_name: str = "gpt-4o-mini" #for openai
    finetuned_model_path: str = "/Users/midhunln/Documents/rag20march_with_eval/finetuned_model" #for finetuned model