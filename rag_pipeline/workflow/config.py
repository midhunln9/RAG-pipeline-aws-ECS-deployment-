"""
Configuration module for RAG Pipeline.

Consolidates all environment-based settings using Pydantic for validation.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults and can be overridden via environment variables.
    """
    
    # Database Configuration
    database_url: str = "sqlite:///sample.db"
    
    # LLM Configuration
    llm_model_name: str = "llama3.2"
    openai_model_name_4o_mini: str = "gpt-4o-mini"
    openai_model_name_5_mini: str = "gpt-5-mini"
    finetuned_model_path: str = "/Users/midhunln/Documents/rag20march_with_eval/finetuned_model"
    
    # Pinecone Configuration
    pinecone_index_name: str = "final-rag-index-openai-small"
    pinecone_metric: str = "dotproduct"
    pinecone_batch_size: int = 200
    pinecone_dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pinecone_sparse_embedding_model: str = "naver/splade-cocondenser-ensembledistil"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # Vector DB Configuration
    pinecone_api_key: str = ""
    pinecone_environment: str = "production"
    
    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"


def get_settings() -> Settings:
    """Get singleton settings instance."""
    return Settings()
