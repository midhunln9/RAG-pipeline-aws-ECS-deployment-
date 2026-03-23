import os
from rag_pipeline.workflow.configs.db_config import DBConfig
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.configs.llm_config import LLMConfig
from rag_pipeline.workflow.configs.vector_db_config import VectorDBConfig
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.database.db_repositories.conversation_repository import ConversationRepository
from rag_pipeline.workflow.repositories.pinecone_repository import PineconeRepository
from rag_pipeline.workflow.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from rag_pipeline.workflow.embeddings.sparse_embedding import SentenceTransformerSparseEmbedding
from rag_pipeline.workflow.llms.ollama_llama import OllamaLLM
from rag_pipeline.workflow.node_orchestrator import Nodes
from rag_pipeline.workflow.graph import RAGWorkflow


class DependencyContainer:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DependencyContainer._initialized:
            self._initialize_dependencies()
            DependencyContainer._initialized = True

    def _initialize_dependencies(self):
        db_config = DBConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite:///sample.db")
        )
        self.database = Database(db_config)

        pinecone_config = PineconeConfig()
        vector_db_config = VectorDBConfig(
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "production")
        )
        llm_config = LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", "llama3.2")
        )

        dense_embedding_strategy = SentenceTransformerEmbedding(pinecone_config)
        sparse_embedding_strategy = SentenceTransformerSparseEmbedding(pinecone_config)

        self.vector_db = PineconeRepository(
            api_key=vector_db_config.api_key,
            environment=vector_db_config.environment,
            pinecone_config=pinecone_config,
            dense_embedding_strategy=dense_embedding_strategy,
            sparse_embedding_strategy=sparse_embedding_strategy
        )

        self.llm = OllamaLLM(llm_config)
        self.conversation_repository = ConversationRepository()

        nodes = Nodes(
            database=self.database,
            vector_db=self.vector_db,
            conversation_repository=self.conversation_repository,
            llm=self.llm
        )

        self.workflow = RAGWorkflow(nodes=nodes)

    def get_workflow(self):
        return self.workflow


def get_dependency_container():
    return DependencyContainer()
