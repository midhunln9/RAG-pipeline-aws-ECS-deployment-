from typing import Protocol
from typing import List
from langchain_core.documents import Document

class VectorDBProtocol(Protocol):
    def query(self, query: str) -> List[Document]:
        """
        Query the vector database for documents.
        """
        ...

