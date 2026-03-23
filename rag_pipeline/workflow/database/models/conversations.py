from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from rag_pipeline.workflow.database.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    messages = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)