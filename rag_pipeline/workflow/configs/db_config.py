from dataclasses import dataclass

@dataclass
class DBConfig:
    database_url: str = "sqlite:///sample.db"