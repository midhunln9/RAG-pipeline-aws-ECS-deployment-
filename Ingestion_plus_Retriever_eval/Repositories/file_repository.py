"""
File Repository module for document access.

This module provides the FileRepository class that implements the
DocumentRepositoryProtocol for loading documents from the local filesystem.
"""

import logging
import os
from typing import List
from Protocols.document_repository_protocol import DocumentRepositoryProtocol


class FileRepository(DocumentRepositoryProtocol):
    """
    Repository for accessing documents stored in the local filesystem.

    This class implements the DocumentRepositoryProtocol to provide
    access to PDF documents stored in a specified directory.

    Attributes:
        file_location (str): Path to the directory containing documents.
        logger (logging.Logger): Logger instance for repository operations.
    """

    def __init__(self, file_location: str = "Ingestion_plus_Retriever_eval/Documents"):
        """
        Initialize the FileRepository with a document location.

        Args:
            file_location (str): Path to the directory containing PDF documents.
                Defaults to "Ingestion_plus_Retriever_eval/Documents".
        """
        self.file_location = file_location
        self.logger = logging.getLogger("DocumentIngestion")
        self.logger.info(f"FileRepository initialized with location: {file_location}")

    def get_documents(self) -> List[str]:
        """
        Retrieve a list of PDF document paths from the file location.

        This method scans the configured directory and returns paths to
        all files with the .pdf extension.

        Returns:
            List[str]: A list of file paths to PDF documents.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        self.logger.debug(f"Scanning directory for documents: {self.file_location}")

        if not os.path.exists(self.file_location):
            self.logger.error(f"Directory not found: {self.file_location}")
            raise FileNotFoundError(f"Directory not found: {self.file_location}")

        files = os.listdir(self.file_location)
        pdf_files = [
            os.path.join(self.file_location, f) for f in files
            if f.endswith('.pdf')
        ]

        self.logger.info(f"Found {len(pdf_files)} PDF documents")
        self.logger.debug(f"Document paths: {pdf_files}")
        return pdf_files
