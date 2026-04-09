"""
LangGraph workflow definition for RAG pipeline.

Defines the state machine topology and node connections for the RAG workflow.
"""

from langgraph.graph import END, START, StateGraph

from rag_pipeline.workflow.node_orchestrator import Nodes
from rag_pipeline.workflow.state import AgentState


class RAGWorkflow:
    """
    LangGraph state machine for RAG pipeline execution.
    
    Defines the topology of the RAG workflow and compiles it into an executable graph.
    """

    def __init__(self, nodes: Nodes):
        """
        Initialize workflow with nodes.
        
        Args:
            nodes: Nodes instance containing graph node implementations.
        """
        self.nodes = nodes
        self.app = self._build_graph()

    def _build_graph(self):
        """
        Build and compile the LangGraph state machine.
        
        Returns:
            Compiled LangGraph application.
        """
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("query_rewriter", self.nodes.query_rewriter)
        graph.add_node("fetch_documents", self.nodes.fetch_documents)
        graph.add_node(
            "generate_summary_last_5_messages",
            self.nodes.generate_summary_last_5_messages,
        )
        graph.add_node("llm_call", self.nodes.llm_call)
        graph.add_node("add_conversation", self.nodes.add_conversation_to_db)

        # Define edges - parallelize independent nodes
        # query_rewriter and generate_summary run in parallel from START
        # fetch_documents depends on query_rewriter (needs rewritten_query)
        # llm_call waits for both fetch_documents and generate_summary (fan-in)
        graph.add_edge(START, "query_rewriter")
        graph.add_edge(START, "generate_summary_last_5_messages")
        graph.add_edge("query_rewriter", "fetch_documents")
        graph.add_edge("fetch_documents", "llm_call")
        graph.add_edge("generate_summary_last_5_messages", "llm_call")
        graph.add_edge("llm_call", "add_conversation")
        graph.add_edge("add_conversation", END)

        return graph.compile()

    def execute(self, query: str, session_id: str) -> dict:
        """
        Execute the RAG workflow with the given query and session.
        
        Args:
            query: User query.
            session_id: User session identifier.
            
        Returns:
            Final workflow output containing response and metadata.
        """
        initial_state = AgentState(query=query, session_id=session_id)
        return self.app.invoke(initial_state)