"""
Repository for conversation persistence.

Handles CRUD operations for conversation history stored in the database.
"""

from sqlalchemy.orm import Session

from rag_pipeline.workflow.database.models.conversations import Conversation


class ConversationRepository:
    """
    Repository for managing conversation records in the database.
    
    Provides methods to store and retrieve conversation history.
    """

    def add_conversation(
        self, session: Session, session_id: str, messages: str
    ) -> None:
        """
        Add a new conversation record to the database.
        
        Args:
            session: SQLAlchemy session for the transaction.
            session_id: Unique identifier for the user session.
            messages: Serialized conversation messages as a string.
        """
        conversation = Conversation(session_id=session_id, messages=messages)
        session.add(conversation)

    def get_conversations_by_session_id(
        self, session: Session, session_id: str, limit: int = 10
    ) -> list[Conversation]:
        """
        Retrieve recent conversations for a given session ID.

        Results are ordered by most recent first, limited to avoid unbounded queries.

        Args:
            session: SQLAlchemy session for the transaction.
            session_id: Unique identifier for the user session.
            limit: Maximum number of conversations to retrieve.

        Returns:
            List of Conversation objects for the session, ordered by recency.
        """
        return (
            session.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .all()
        )