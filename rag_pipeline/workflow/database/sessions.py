from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from rag_pipeline.workflow.configs.db_config import DBConfig
from rag_pipeline.workflow.database.base import Base
from rag_pipeline.workflow.database.models.conversations import Conversation
from typing import List, Dict, Any

class Database:
    def __init__(self, db_config: DBConfig):
        self.engine = create_engine(db_config.database_url, echo=False, future=True)
        self.session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(self.engine)
        
    @contextmanager
    def session_scope(self):
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    

            

