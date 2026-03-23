from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any
import json

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    messages = Column(String)
    created_at = Column(DateTime, default=datetime.now)


class Database:
    def __init__(self):
        self.client = create_engine("sqlite:///sample.db", echo=False)
        self.session = sessionmaker(bind=self.client)

    @contextmanager
    def session_scope(self):
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def create_table(self):
        Base.metadata.create_all(self.client)

    def add_conversation(self, session_id: str, messages: List[Dict[str, Any]]):
        with self.session_scope() as session:
            query = text("""
                INSERT INTO conversations (session_id, messages, created_at)
                VALUES (:session_id, :messages, :created_at)
            """)
            session.execute(
                query,
                {
                    "session_id": session_id,
                    "messages": json.dumps(messages),
                    "created_at": datetime.now(),
                },
            )

    def get_conversations(self, session_id: str = None):
        with self.session_scope() as session:
            if session_id:
                query = text("""
                    SELECT id, session_id, messages, created_at
                    FROM conversations
                    WHERE session_id = :session_id
                    ORDER BY created_at DESC
                """)
                result = session.execute(query, {"session_id": session_id})
            else:
                query = text("""
                    SELECT id, session_id, messages, created_at
                    FROM conversations
                    ORDER BY created_at DESC
                """)
                result = session.execute(query)

            conversations = []
            for row in result.fetchall():
                conversations.append(
                    {
                        "id": row.id,
                        "session_id": row.session_id,
                        "messages": json.loads(row.messages),
                        "created_at": row.created_at,
                    }
                )

            return conversations

# EXAMPLE USAGE

db = Database()
db.create_table()

db.add_conversation(
    session_id="abc123",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ],
)

all_conversations = db.get_conversations()
print(all_conversations)

filtered_conversations = db.get_conversations(session_id="abc123")
print(filtered_conversations)