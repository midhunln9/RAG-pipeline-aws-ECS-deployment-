"""
Splitter Strategy module.

This module defines the SplitterStrategy abstract base class that
provides a common interface for all document splitting strategies.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class SplitterStrategy(ABC):
    """
    Abstract base class for document splitting strategies.

    This class defines the interface that all text splitting strategies
    must implement. It provides a standard contract for breaking documents
    into smaller chunks using various algorithms (recursive character
    splitting, token-based splitting, semantic splitting, etc.).

    Implementations of this class should handle:
    - Splitting documents according to their specific algorithm
    - Maintaining document metadata during splitting
    - Respecting semantic boundaries where applicable

    Example:
        >>> class MySplitter(SplitterStrategy):
        ...     def split_documents(self, documents: List[Document]) -> List[Document]:
        ...         # Implementation here
        ...         return chunks
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        This abstract method must be implemented by all concrete
        splitting strategies to define their specific document
        chunking logic.

        Args:
            documents (List[Document]): A list of LangChain Document objects
                to be split into chunks.

        Returns:
            List[Document]: A list of document chunks, where each chunk
                is a LangChain Document object with preserved metadata.

        Raises:
            NotImplementedError: If the concrete class does not
                implement this method.
        """
        ...
