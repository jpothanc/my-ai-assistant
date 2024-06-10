# My personal AI assistant using RAG and ChromaDB
I have been preparing for the az-900 exam and I wanted to create a personal AI assistant that can help me with my preparation. 
I have used the RAG model from huggingface and ChromaDB to create this assistant.

# setup
Python version 3.12.0
Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

## Install dependencies
```python
pip install -r requirements.txt
pip install "unstructured[md]"
```

## Retrieve the data
* Load the data from the markdown files (azure documentation, prepared for the az-900 exam)
* Split the data into paragraphs
* Load into the Chroma DB
* Save the data

```python
python retrieve_data.py
```

## Query the database

* Ask a question.
* Get the possible responses from chroma db.
* feed in these responses to openai to get the best response.

```python
python query_data.py
```

## What is ChromaDB?
ChromaDB is a specialized database designed to handle high-dimensional vectors. Vectors are mathematical representations of data, often used in machine learning and artificial intelligence to capture the meaning or characteristics of the data. ChromaDB helps store, manage, and search through these vectors efficiently.

