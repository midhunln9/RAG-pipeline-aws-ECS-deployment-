"""
Ask endpoint for RAG pipeline.

Provides HTTP interface to query the RAG pipeline and get responses.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Router will be imported in main.py and registered with the app
router = APIRouter()


class AskRequest(BaseModel):
    """Request model for the ask endpoint."""

    query: str
    session_id: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is financial compliance?",
                "session_id": "6498"
            }
        }
    }


class SourceDocument(BaseModel):
    """Model for a source document in the response."""

    content: str
    metadata: dict


class AskResponse(BaseModel):
    """Response model from the ask endpoint."""

    response: str
    session_id: str
    sources: list[SourceDocument] = []


@router.post("/ask", response_model=AskResponse)
def ask(request: Request, ask_request: AskRequest) -> AskResponse:
    """
    Query the RAG pipeline and get a response.
    
    Args:
        request: The FastAPI request object containing app.state.
        ask_request: The ask request containing query and session_id.
        
    Returns:
        AskResponse containing the generated response and source documents.
        
    Raises:
        HTTPException: If processing fails.
    """
    if not hasattr(request.app.state, 'workflow'):
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        logger.info(
            f"Processing query for session {ask_request.session_id}: {ask_request.query[:50]}..."
        )
        
        # Execute the workflow
        result = request.app.state.workflow.execute(
            query=ask_request.query,
            session_id=ask_request.session_id
        )
        
        logger.debug(f"Workflow execution result keys: {result.keys()}")
        logger.debug(f"Response field present: {'response' in result}")
        
        response_value = result.get("response")
        logger.info(f"Response value type: {type(response_value)}")
        logger.info(f"Response value length: {len(response_value) if response_value else 0}")
        logger.info(f"Response value preview: {response_value[:100] if response_value else 'None'}")

        # Extract sources from retrieved documents
        sources = []
        if result.get("retrieved_documents"):
            for doc in result["retrieved_documents"]:
                sources.append(
                    SourceDocument(
                        content=doc.page_content,
                        metadata=doc.metadata if doc.metadata else {}
                    )
                )
        
        logger.info(f"Extracted {len(sources)} source documents")

        return AskResponse(
            response=response_value if response_value else "I don't know.",
            session_id=ask_request.session_id,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query")