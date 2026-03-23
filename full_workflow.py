from typing import TypedDict, Annotated, Sequence
from langchain_core.documents import Document
from typing import List, Dict
from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from workflow.repository.pinecone_repository import PineconeRepository
from configs.pinecone_config import PineconeConfig
from dotenv import load_dotenv
import os
from workflow.sparse_embedding import SentenceTransformerSparseEmbedding
from workflow.dense_embedding import SentenceTransformerEmbedding
from langgraph.graph.message import add_messages
from workflow.database.sessions import Database
from workflow.database.db_repositories.conversation_repository import ConversationRepository
from workflow.database.configs.db_config import DBConfig


load_dotenv("/Users/midhunln/Documents/rag20march_with_eval/Ingestion_plus_Retriever_eval/ingestion.env")

pinecone_config = PineconeConfig()

pinecone_repository = PineconeRepository(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"),
        dense_embedding_strategy=SentenceTransformerEmbedding(pinecone_config),
        sparse_embedding_strategy=SentenceTransformerSparseEmbedding(pinecone_config),
        pinecone_config=pinecone_config
    )

db_config = DBConfig()
database = Database(db_config)
conversation_repository = ConversationRepository()



llm = ChatOllama(model="llama3.2", temperature=0)


class AgentState(TypedDict):
    """
    Represents the state of an agent.
    """
    query: str
    rewritten_query : str
    retrieved_documents : List[Document]
    conversation_history : Annotated[Sequence[BaseMessage], add_messages]
    response : str

graph = StateGraph(AgentState)


def query_rewriter(state: AgentState) -> Dict:
    """
    Rewrite the query to be more specific.
    """
    prompt = "You are a helpful assistant that rewrites queries given by the user."
    messages = [HumanMessage(content=state["query"])]
    response = llm.invoke(messages)
    return {"rewritten_query": response.content}


def add_conversation(state: AgentState) -> Dict:
    """
    Add the conversation to the database.
    """

    with database.session_scope() as session:
        conversation_repository.add_conversation(session, state["session_id"], state["query"])
    return {}

def retrieve_documents(state: AgentState) -> Dict:
    """
    Retrieve documents from the Pinecone repository.
    """
    documents = pinecone_repository.query(state["query"])
    return {"retrieved_documents": documents}

def augment_query_and_rag(state: AgentState) -> Dict:
    """
    Augment the query to be more specific.
    """
    messages = [HumanMessage(content=state["rewritten_query"]), 
    SystemMessage(content="You are a helpful assistant that retrieves documents from the Pinecone repository.")]
    response = llm.invoke(messages)
    return {"query": response.content}
