from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.database.models.conversations import Conversation
from sqlalchemy.orm import Session
from typing import List
from rag_pipeline.workflow.protocols.database_repo_protocol import DatabaseRepositoryProtocol

class ConversationRepository(DatabaseRepositoryProtocol):
    def add_conversation(self, session: Session, session_id: str, messages: str) -> None:
        conversation = Conversation(session_id=session_id, messages=messages)
        session.add(conversation)

    def get_conversations_by_session_id(self, session: Session, session_id: str) -> List[Conversation]:
        return (
            session.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .all()
        )