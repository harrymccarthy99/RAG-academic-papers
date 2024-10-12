from langchain_community.document_loaders import DirectoryLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import openai 
from dotenv import load_dotenv # This is needed is API keys in a env file
import os
import shutil
import faiss
from uuid import uuid4

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

path_to_faiss_index = "faiss"
path_to_data = "data"


def main():
    generate_vector_store()


def generate_vector_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)


def load_documents():
    loader = DirectoryLoader(path_to_data, glob = "*.pdf")
    documents = loader.load()
    return documents


def split_text(documents):
    
    # Defining the recursive splitter
    embedding_model = OpenAIEmbeddings()
    semantic_splitter = SemanticChunker(embeddings= embedding_model, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.8)

    
    # Getting chunks with text splitter
    chunks = semantic_splitter.split_documents(documents)
    
    # Printing number of original documents and number of chunks
    print(f'Number of original document: {len(documents)} | Number of chunks; {len(chunks)}')
    
    # Printing an example of chunk content and metadata
    print("--- Example ---")
    print(f'Content; {chunks[2].page_content}')
    print(f'Metadata; {chunks[2].metadata}')

    return chunks


def save_to_faiss(chunks):
    # Defining the recursive splitter
    embedding_model = OpenAIEmbeddings()

    # Delete old database
    if os.path.exists(path_to_faiss_index):
        os.remove(path_to_faiss_index) 
    
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("placeholder")))
    
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        )
    
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    vector_store.save_local(path_to_faiss_index)
    
    # Creating a new DB from the documents
    # db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory = path_to_chroma)

    # # Database should save automatically but we can force save using persist

    # # Creating a new FAISS index from the documents
    # db = FAISS.from_documents(chunks, OpenAIEmbeddings())

    # # Save the FAISS index manually to a file
    # faiss.write_index(db.index, path_to_faiss_index)

    print(f'Saved {len(chunks)} chunks to {path_to_faiss_index}')



if __name__ == "__main__":
    main()