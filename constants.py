CHROMA_DB_PATH = "chroma"
DATA_PATH = "data/azure"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
