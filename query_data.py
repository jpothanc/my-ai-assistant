from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from constants import CHROMA_DB_PATH, PROMPT_TEMPLATE


# Query the Chroma DB with the user input.
# If the similarity score is less than 0.7, then we will not return any results.
# Pass the context text and the question to the ChatOpenAI model and get the response.
def query_chroma(query_text):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)
    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}"
    print(formatted_response)


def main():
    while True:
        user_input = input("Enter your question:").lower()
        if user_input == "exit":
            break
        query_text = user_input
        query_chroma(query_text)


if __name__ == "__main__":
    main()
