"""
FastAPI application for RAG Pipeline.

Uses async context manager for dependency initialization and lifecycle management.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from rag_pipeline.workflow.config import Settings
from rag_pipeline.workflow.service import RAGService
from rag_pipeline.workflow.graph import RAGWorkflow
from rag_pipeline.workflow.node_orchestrator import Nodes
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.repositories.pinecone_repository import PineconeRepository
from rag_pipeline.workflow.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from rag_pipeline.workflow.embeddings.sparse_embedding import (
    SentenceTransformerSparseEmbedding,
)
from rag_pipeline.workflow.llms.ollama_llama import OllamaLLM
from rag_pipeline.workflow.database.db_repositories.conversation_repository import (
    ConversationRepository,
)
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.configs.llm_config import LLMConfig
from rag_pipeline.api.routes import ask_endpoint
from rag_pipeline.workflow.llms.openai import OpenAILLM
from rag_pipeline.workflow.llms.finetuned_llm import FinetunedLLM
from rag_pipeline.workflow.embeddings.openai_embedding import OpenAIEmbedding
from dotenv import load_dotenv, find_dotenv

load_dotenv("/Users/midhunln/Documents/rag20march_with_eval/Ingestion_plus_Retriever_eval/ingestion.env")
"""
load_dotenv(find_dotenv())
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_dependencies() -> dict:
    """
    Validate that all required dependencies and API keys are available.
    
    Returns:
        Dictionary with validation results and any error messages.
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        validation_results["valid"] = False
        validation_results["errors"].append("OPENAI_API_KEY not found in environment")
    
    settings = Settings()
    if not settings.pinecone_api_key or settings.pinecone_api_key == "":
        validation_results["valid"] = False
        validation_results["errors"].append("PINECONE_API_KEY not configured in settings")
    
    if not settings.database_url:
        validation_results["valid"] = False
        validation_results["errors"].append("DATABASE_URL not configured")
    
    return validation_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown.
    
    On startup:
    - Validates all dependencies
    - Initializes all dependencies (database, embeddings, vector DB, LLM, service, workflow)
    - Stores them in app.state for access from route handlers
    
    On shutdown:
    - Cleans up resources
    """
    logger.info("Starting up RAG Pipeline API...")
    
    validation = validate_dependencies()
    if not validation["valid"]:
        for error in validation["errors"]:
            logger.error(f"STARTUP VALIDATION ERROR: {error}")
        logger.error("Startup failed due to missing dependencies")
        raise RuntimeError(f"Startup validation failed: {'; '.join(validation['errors'])}")
    
    for warning in validation["warnings"]:
        logger.warning(f"STARTUP WARNING: {warning}")
    
    logger.info("All dependencies validated successfully")
    
    # Initialize settings
    settings = Settings()
    app.state.settings = settings
    logger.info(f"Loaded settings for environment: {settings.environment}")

    # Database
    database = Database(settings.database_url)
    app.state.database = database
    logger.info("Database initialized")

    # Embeddings configuration - use settings to ensure alignment
    pinecone_config = PineconeConfig.from_settings(settings)
    logger.info(f"Pinecone config: index={pinecone_config.index_name}, metric={pinecone_config.metric}")

    # Embedding strategies
    """
    dense_embedding = SentenceTransformerEmbedding(pinecone_config)
    """
    dense_embedding = OpenAIEmbedding()
    sparse_embedding = SentenceTransformerSparseEmbedding(pinecone_config)
    app.state.dense_embedding = dense_embedding
    app.state.sparse_embedding = sparse_embedding
    logger.info("Embedding strategies initialized")

    # Vector Database
    vector_db = PineconeRepository(
        api_key=settings.pinecone_api_key,
        pinecone_config=pinecone_config,
        dense_embedding_strategy=dense_embedding,
        sparse_embedding_strategy=sparse_embedding,
        environment=settings.pinecone_environment,
    )
    app.state.vector_db = vector_db
    logger.info("Vector database initialized")

    # LLM
    llm_config = LLMConfig(model_name=settings.llm_model_name, openai_model_name=settings.openai_model_name)
    """
    llm = OllamaLLM(llm_config)
    """
    llm = OpenAILLM(llm_config)
    """
    llm = FinetunedLLM(llm_config)
    """
    app.state.llm = llm
    logger.info("LLM initialized")

    # Repository
    conversation_repo = ConversationRepository()
    app.state.conversation_repo = conversation_repo
    logger.info("Conversation repository initialized")

    # Service
    service = RAGService(
        database=database,
        vector_db=vector_db,
        conversation_repository=conversation_repo,
        llm=llm,
    )
    app.state.service = service
    logger.info("RAG service initialized")

    # Workflow
    nodes = Nodes(service=service)
    workflow = RAGWorkflow(nodes=nodes)
    app.state.workflow = workflow
    logger.info("Workflow initialized and compiled")
    
    logger.info("RAG Pipeline API startup complete")
    
    yield
    
    logger.info("Shutting down RAG Pipeline API...")
    logger.info("Cleanup complete")


# Create FastAPI app with lifespan context manager
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# Register router
app.include_router(ask_endpoint.router)


@app.get("/")
def health():
    """Health check endpoint."""
    return {
        "message": "RAG Pipeline API is running",
        "status": "healthy",
    }