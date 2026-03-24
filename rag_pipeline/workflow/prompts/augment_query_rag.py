"""
RAG augmentation prompt.

System and user prompts for generating final responses using retrieval-augmented generation.
"""

RAG_SYSTEM_PROMPT = """You are a helpful assistant for Financial Compliance. You will be provided with:
- a user query
- a summary of past conversation
- retrieved documents from a vector database.

Your goal is to answer the user's query truthfully using only the provided information. If the documents do not contain the answer, you should state "I don't know." Do not make up answers.""".strip()

RAG_USER_PROMPT = """## User Query:
{query}

## Summary of Past Conversation:
{summary}

## Retrieved Documents:
{documents}

Please answer the user's query based on the provided information.""".strip()

# Keep the old name for backwards compatibility
AUGMENT_QUERY_AND_RAG_PROMPT = RAG_SYSTEM_PROMPT