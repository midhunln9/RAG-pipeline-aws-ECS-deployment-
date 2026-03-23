from typing import Protocol
from sqlalchemy.orm import Session

class DatabaseRepositoryProtocol(Protocol):
    def add_conversation(self, session: Session, session_id: str, messages: str) -> None:
        pass

    def get_conversations_by_session_id(self, session: Session, session_id: str) -> List[Conversation]:
        pass