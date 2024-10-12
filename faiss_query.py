import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import openai
from dotenv import load_dotenv # This is needed is API keys in a env file
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
import numpy as np

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.

openai.api_key = os.environ['OPENAI_API_KEY']

path_to_faiss_index = "faiss"

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

    embedding_model = OpenAIEmbeddings()  # Adjust as necessary

    if os.path.exists(path_to_faiss_index):
        vector_store = FAISS.load_local(path_to_faiss_index, embedding_model, allow_dangerous_deserialization=True)

    else:
         print("No FAISS index found")
    
    # Search the DB.
    results = vector_store.similarity_search_with_score(query_text, k=3)

    if len(results) == 0 or results[0][1] > 0.7:
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