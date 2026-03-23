from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    database_url: str = "sqlite:///sample.db"
