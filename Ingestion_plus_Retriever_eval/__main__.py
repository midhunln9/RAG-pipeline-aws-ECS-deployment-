"""
Main entry point for the document ingestion pipeline.

This module initializes and runs the document processing pipeline,
including document loading, chunking, and text splitting.
"""

from src.pipeline import Pipeline
from src.chunker_service import ChunkerService
from Repositories.file_repository import FileRepository
from src.recursive_character_text_splitting import RecursiveCharacterTextSplitting
from configs.recursive_text_splitter_config import RecursiveCharacterTextSplittingConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import setup_logging


def main():
    """
    Main function to initialize and run the document ingestion pipeline.

    This function sets up logging, creates all necessary components
    (file repository, text splitter, chunker service), and executes
    the pipeline to process documents.

    Returns:
        list: A list of document chunks extracted from the processed documents.
    """
    logger = setup_logging()
    logger.info("Starting document ingestion pipeline")

    logger.info("Initializing file repository")
    file_repository = FileRepository()

    logger.info("Configuring text splitting parameters")
    config = RecursiveCharacterTextSplittingConfig(
        chunk_size=1000,
        chunk_overlap=200
    )
    logger.debug(f"Chunk size: {config.chunk_size}, Chunk overlap: {config.chunk_overlap}")

    logger.info("Initializing recursive character text splitter")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    splitting_strategy = RecursiveCharacterTextSplitting(config, splitter)

    logger.info("Initializing chunker service")
    chunker_service = ChunkerService(
        document_repository=file_repository,
        splitter_strategy=splitting_strategy
    )

    logger.info("Initializing pipeline")
    pipeline = Pipeline(
        chunker_service=chunker_service,
        document_repository=file_repository
    )

    logger.info("Running pipeline")
    chunks = pipeline.run()

    logger.info(f"Successfully processed {len(chunks)} chunks from documents")
    print(f"Successfully processed {len(chunks)} chunks from documents")
    return chunks


if __name__ == "__main__":
    main()
