"""
Recursive Text Splitter Configuration module.

This module provides configuration dataclasses for the
RecursiveCharacterTextSplitter, allowing customization of
chunking parameters.
"""

from dataclasses import dataclass, field


@dataclass
class RecursiveCharacterTextSplittingConfig:
    """
    Configuration parameters for recursive character text splitting.

    This dataclass holds the configuration settings for LangChain's
    RecursiveCharacterTextSplitter, controlling how documents are
    divided into chunks.

    Attributes:
        chunk_size (int): The maximum number of characters in each chunk.
            Defaults to 1000 characters.
        chunk_overlap (int): The number of characters to overlap between
            consecutive chunks. This helps preserve context at chunk boundaries.
            Defaults to 200 characters.

    Example:
        >>> config = RecursiveCharacterTextSplittingConfig(
        ...     chunk_size=500,
        ...     chunk_overlap=50
        ... )
        >>> print(config.chunk_size)
        500
    """

    chunk_size: int = field(default=1000)
    """Maximum number of characters per chunk."""

    chunk_overlap: int = field(default=200)
    """Number of overlapping characters between chunks."""
