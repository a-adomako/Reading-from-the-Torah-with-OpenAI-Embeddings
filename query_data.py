import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "path/to/chroma"

PROMPT_TEMPLATE = """
Answer the question based on only the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI
    parser = argparse.ArgumentParser(description="Query the Chroma database with a question.")
    parser.add_argument(
        "query_text", 
        type=str, 
        nargs="?",  # Makes the argument optional
        help="The query text. If not provided, you'll be prompted to input it."
    )
    args = parser.parse_args()

    # Handle missing query_text
    if not args.query_text:
        query_text = input("Enter your query: ")
    else:
        query_text = args.query_text

    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Generated Prompt:")
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
