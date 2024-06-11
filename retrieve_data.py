from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import os
import shutil

from constants import DATA_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# assuming that environment called "OPENAI_API_KEY" is set.
# This is your openai api key that you can get from openai.com
openai.api_key = os.getenv("OPENAI_API_KEY")


def retrieve_data():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


# Load my Azure Notes which is in Markdown format.
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


# Split the data into paragraphs.
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


# Chroma is a vector store that can be used to store and retrieve embeddings of text data.
# Save the split chunks in to Chroma DB after converting this in to embeddings.
# We use openai embeddings to convert the text to embeddings.
def save_to_chroma(chunks: list[Document]):
    # Delete the existing DB first.
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_DB_PATH}.")


if __name__ == "__main__":
    retrieve_data()
