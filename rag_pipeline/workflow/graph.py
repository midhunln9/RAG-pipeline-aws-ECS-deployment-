from langgraph.graph import StateGraph, END, START
from rag_pipeline.workflow.state import AgentState
from langchain_core.messages import HumanMessage
from rag_pipeline.workflow.node_orchestrator import Nodes

class RAGWorkflow:
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
        self.app = self.create_graph()

    def create_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("query_rewriter", self.nodes.query_rewriter)
        graph.add_node("fetch_documents", self.nodes.fetch_documents)
        graph.add_node("generate_summary_last_5_messages", self.nodes.generate_summary_last_5_messages)
        graph.add_node("llm_call", self.nodes.llm_call)
        graph.add_node("add_conversation", self.nodes.add_conversation_to_db)

        graph.add_edge(START, "query_rewriter")
        graph.add_edge("query_rewriter", "fetch_documents")
        graph.add_edge("fetch_documents", "generate_summary_last_5_messages")
        graph.add_edge("generate_summary_last_5_messages", "llm_call")
        graph.add_edge("llm_call", "add_conversation")
        graph.add_edge("add_conversation", END)

        return graph.compile()

    def run(self, query: str, session_id: str):
        state = AgentState(query=query, session_id=session_id, conversation_history=[HumanMessage(content=query)])
        return self.app.invoke(state)