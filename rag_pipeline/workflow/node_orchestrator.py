from rag_pipeline.workflow.state import AgentState
from rag_pipeline.workflow.prompts.query_rewriter import QUERY_REWRITER_PROMPT
from rag_pipeline.workflow.prompts.summary_so_far import SUMMARY_SO_FAR
from rag_pipeline.workflow.prompts.augment_query_rag import AUGMENT_QUERY_AND_RAG_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from rag_pipeline.workflow.database.db_repositories.conversation_repository import ConversationRepository
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.protocols.vector_db_protocol import VectorDBProtocol
from rag_pipeline.workflow.protocols.llm_protocol import LLMProtocol
from typing import List

class Nodes:
    def __init__(self, database: Database,
    vector_db: VectorDBProtocol,
    conversation_repository: ConversationRepository,
    llm : LLMProtocol):
        self.database = database
        self.vector_db = vector_db
        self.conversation_repository = conversation_repository
        self.llm = llm
        
    def query_rewriter(self, state: AgentState):
        system_message = SystemMessage(content=QUERY_REWRITER_PROMPT)
        human_message = HumanMessage(content=state["query"])
        response = self.llm.invoke([system_message, human_message])
        return {"rewritten_query": response.content, 
        "conversation_history": [HumanMessage(content=state["query"]), 
        AIMessage(content=response.content)]}
    
    def fetch_documents(self, state: AgentState):
        # over here, i need embedding of the rewritten query and then use the vector db to query the documents
        documents = self.vector_db.query(state["rewritten_query"])
        return {"retrieved_documents": documents}
    
    def generate_summary_last_5_messages(self, state: AgentState):
        try:
            with self.database.session_scope() as session:
                past_conv = self.conversation_repository.get_conversations_by_session_id(session, state["session_id"])
                if len(past_conv) > 5:
                    past_conv = past_conv[-5:] # Get the last 5 conversations
                    summary_so_far = self.llm.invoke([SystemMessage(content=SUMMARY_SO_FAR),
                    HumanMessage(content=str(past_conv))])
                    return {"summary_before_last_five_messages": summary_so_far.content}
                else:
                    return {"summary_before_last_five_messages": "No past conversation summary available."}
        except Exception as e:
            return {"summary_before_last_five_messages": "None"}

    def llm_call(self, state: AgentState):
        system_message = SystemMessage(content=AUGMENT_QUERY_AND_RAG_PROMPT)
        human_message = HumanMessage(content=state["rewritten_query"])
        summary_message = HumanMessage(content=state["summary_before_last_five_messages"])
        docs_str = "\n".join([f"Document: {doc.page_content}\nMetadata: {doc.metadata}" for doc in state["retrieved_documents"]]) if state["retrieved_documents"] else "No documents retrieved"
        retrived_docs_message = HumanMessage(content=docs_str)
        response = self.llm.invoke([system_message, human_message, summary_message, retrived_docs_message])
        return {"response": response.content}


    def add_conversation_to_db(self, state: AgentState):
        with self.database.session_scope() as session:
            # Assuming conversation_history is a list of BaseMessage, we need to serialize it for storage
            serialized_messages = [str(msg) for msg in state["conversation_history"]]
            self.conversation_repository.add_conversation(session, state["session_id"], "\n".join(serialized_messages))
        return {}