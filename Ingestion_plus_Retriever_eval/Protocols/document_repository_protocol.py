"""
Document Repository Protocol module.

This module defines the DocumentRepositoryProtocol interface that all
document repository implementations must follow. It provides a standard
contract for accessing documents from various sources.
"""

from typing import Protocol, List


class DocumentRepositoryProtocol(Protocol):
    """
    Protocol defining the interface for document repositories.

    This protocol establishes the contract that all document repository
    implementations must follow. It provides a standardized way to access
    documents from different sources (filesystem, cloud storage, databases, etc.).

    Implementations of this protocol should handle:
    - Downloading or accessing documents from their source
    - Returning a list of accessible document identifiers/paths
    - Managing document metadata and access patterns
    """

    def get_documents(self) -> List[str]:
        """
        Retrieve a list of document identifiers from the repository.

        This method should return paths, URLs, or identifiers for all
        available documents that can be processed by the ingestion pipeline.

        Returns:
            List[str]: A list of document identifiers (file paths, URLs, etc.).

        Example:
            For a file-based implementation:
            - Download documents into a folder
            - Extract document names/paths from the folder
            - Return list of paths to load documents one by one
        """
        ...
