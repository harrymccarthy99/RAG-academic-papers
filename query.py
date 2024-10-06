import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import openai
from dotenv import load_dotenv # This is needed is API keys in a env file
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

path_to_chroma = "chroma"

prompt_template =ChatPromptTemplate.from_template("""
Use the following piece of context to answer the question at the end.
If you don't know the answer, say that you don't know
Context: {context}
Question: {question}
""")


def main():
    # Create CLI

    # This allows us to enter the query text in the command line
    parser = argparse.ArgumentParser()
    # "query_text" is the name of the argument the user will pass in the command line
    parser.add_argument("query_text", type=str, help="The query text.")
    # This line processes the command line arguments and stores them in args
    args = parser.parse_args()
    # The arguments are stored as attributes of args
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=path_to_chroma, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()