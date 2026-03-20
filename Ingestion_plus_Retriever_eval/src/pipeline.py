"""
Pipeline module for document ingestion and processing.

This module contains the Pipeline class that orchestrates the document
chunking process by coordinating between the chunker service and
document repository.
"""

import logging
from Protocols.document_repository_protocol import DocumentRepositoryProtocol
from src.chunker_service import ChunkerService


class Pipeline:
    """
    Orchestrates the document ingestion and chunking process.

    This class coordinates between the chunker service and document repository
    to load, process, and split documents into manageable chunks.

    Attributes:
        chunker_service (ChunkerService): Service responsible for chunking documents.
        document_repository (DocumentRepositoryProtocol): Repository for accessing documents.
        logger (logging.Logger): Logger instance for pipeline operations.
    """

    def __init__(self,
                 chunker_service: ChunkerService,
                 document_repository: DocumentRepositoryProtocol):
        """
        Initialize the Pipeline with required services.

        Args:
            chunker_service (ChunkerService): Service responsible for chunking documents.
            document_repository (DocumentRepositoryProtocol): Repository for accessing documents.
        """
        self.chunker_service = chunker_service
        self.document_repository = document_repository
        self.logger = logging.getLogger("DocumentIngestion")
        self.logger.debug("Pipeline initialized with chunker_service and document_repository")

    def run(self):
        """
        Execute the document processing pipeline.

        This method triggers the chunking process through the chunker service
        and returns the resulting document chunks.

        Returns:
            list: A list of document chunks extracted from processed documents.
        """
        self.logger.info("Starting pipeline execution")
        chunks = self.chunker_service.chunk_documents()
        self.logger.info(f"Pipeline execution completed. Generated {len(chunks)} chunks")
        return chunks
