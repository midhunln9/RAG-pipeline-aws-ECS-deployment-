"""
Chunker Service module for document processing.

This module provides the ChunkerService class that handles loading documents
from various sources and splitting them into chunks using a configured
splitting strategy.
"""

import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from Protocols.document_repository_protocol import DocumentRepositoryProtocol
from strategy.splitter_strategy import SplitterStrategy


class ChunkerService:
    """
    Service responsible for loading and chunking documents.

    This class handles the document loading process from the repository
    and applies text splitting strategies to break documents into
    manageable chunks for further processing.

    Attributes:
        documents (list): List of document file paths from the repository.
        splitter_strategy (SplitterStrategy): Strategy for splitting documents into chunks.
        logger (logging.Logger): Logger instance for chunker operations.
    """

    def __init__(self, document_repository: DocumentRepositoryProtocol,
                 splitter_strategy: SplitterStrategy):
        """
        Initialize the ChunkerService with repository and splitting strategy.

        Args:
            document_repository (DocumentRepositoryProtocol): Repository for accessing documents.
            splitter_strategy (SplitterStrategy): Strategy for splitting documents into chunks.
        """
        self.documents = document_repository.get_documents()
        self.splitter_strategy = splitter_strategy
        self.logger = logging.getLogger("DocumentIngestion")
        self.logger.info(f"ChunkerService initialized with {len(self.documents)} documents")

    def chunk_documents(self) -> List[Document]:
        """
        Load and chunk all documents from the repository.

        This method loads each document using PyPDFLoader, extracts content,
        and applies the configured splitting strategy to create chunks.

        Returns:
            List[Document]: A list of document chunks extracted from all documents.
        """
        self.logger.info("Starting document chunking process")
        chunks = []

        for document_path in self.documents:
            self.logger.debug(f"Loading document: {document_path}")
            try:
                loader = PyPDFLoader(document_path, mode="page")
                documents = loader.load()
                self.logger.info(f"Loaded {len(documents)} pages from {document_path}")

                document_chunks = self.splitter_strategy.split_documents(documents)
                self.logger.info(f"Split into {len(document_chunks)} chunks")
                chunks.extend(document_chunks)
            except Exception as e:
                self.logger.error(f"Error processing document {document_path}: {str(e)}")
                raise

        self.logger.info(f"Chunking completed. Total chunks generated: {len(chunks)}")
        return chunks
