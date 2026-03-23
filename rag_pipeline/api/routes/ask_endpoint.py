from fastapi import APIRouter
from pydantic import BaseModel
from rag_pipeline.api.dependencies import get_dependency_container

ask_endpoint = APIRouter()


class AskRequest(BaseModel):
    query: str
    session_id: str


@ask_endpoint.post("/ask")
async def ask(request: AskRequest):
    container = get_dependency_container()
    workflow = container.get_workflow()
    result = workflow.run(query=request.query, session_id=request.session_id)
    return {"result": result}

