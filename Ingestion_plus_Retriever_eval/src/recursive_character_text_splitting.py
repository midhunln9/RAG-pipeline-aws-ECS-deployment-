"""
Recursive Character Text Splitting strategy implementation.

This module provides a concrete implementation of the SplitterStrategy
using LangChain's RecursiveCharacterTextSplitter for intelligent
document chunking based on character boundaries.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from strategy.splitter_strategy import SplitterStrategy
from configs.recursive_text_splitter_config import RecursiveCharacterTextSplittingConfig


class RecursiveCharacterTextSplitting(SplitterStrategy):
    """
    Text splitting strategy using recursive character-based splitting.

    This class implements the SplitterStrategy interface using LangChain's
    RecursiveCharacterTextSplitter, which splits text recursively by
    different separators (paragraphs, sentences, words) to create
    semantically meaningful chunks.

    Attributes:
        config (RecursiveCharacterTextSplittingConfig): Configuration for chunking parameters.
        splitter (RecursiveCharacterTextSplitter): LangChain text splitter instance.
        logger (logging.Logger): Logger instance for splitting operations.
    """

    def __init__(self, config: RecursiveCharacterTextSplittingConfig,
                 splitter: RecursiveCharacterTextSplitter):
        """
        Initialize the RecursiveCharacterTextSplitting strategy.

        Args:
            config (RecursiveCharacterTextSplittingConfig): Configuration containing
                chunk_size and chunk_overlap parameters.
            splitter (RecursiveCharacterTextSplitter): Pre-configured LangChain
                text splitter instance.
        """
        self.config = config
        self.splitter = splitter
        self.logger = logging.getLogger("DocumentIngestion")
        self.logger.debug(
            f"RecursiveCharacterTextSplitting initialized with "
            f"chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap}"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        Uses the configured RecursiveCharacterTextSplitter to break documents
        into smaller pieces while respecting semantic boundaries.

        Args:
            documents (List[Document]): List of LangChain Document objects to split.

        Returns:
            List[Document]: A list of document chunks.
        """
        self.logger.debug(f"Splitting {len(documents)} documents into chunks")
        chunks = self.splitter.split_documents(documents)
        self.logger.debug(f"Document splitting completed: {len(chunks)} chunks generated")
        return chunks
