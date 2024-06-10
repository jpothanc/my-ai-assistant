# My personal AI assistant using RAG and ChromaDB

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
python query_data.py "What is iaas, paas and saas?"
```

What is ChromaDB?
ChromaDB is a specialized database designed to handle high-dimensional vectors. Vectors are mathematical representations of data, often used in machine learning and artificial intelligence to capture the meaning or characteristics of the data. ChromaDB helps store, manage, and search through these vectors efficiently.

