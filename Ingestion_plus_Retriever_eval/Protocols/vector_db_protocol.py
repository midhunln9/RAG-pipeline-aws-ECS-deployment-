from typing import Protocol, List
from langchain_core.documents import Document

class VectorDBProtocol(Protocol):
    def upsert_chunks(self, chunks: List[Document]):
        ...
    def query_vector_store_for_rankx(self, query : str) -> List[Document]:
        ...