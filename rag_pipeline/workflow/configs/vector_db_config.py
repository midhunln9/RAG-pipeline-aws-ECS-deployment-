from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    api_key: str = ""
    environment: str = "production"
