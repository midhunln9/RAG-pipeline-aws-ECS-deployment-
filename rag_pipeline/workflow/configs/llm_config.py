from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name : str = "llama3.2"