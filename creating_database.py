from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv # This is needed is API keys in a env file
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

path_to_chroma = "chroma"
path_to_data = "data/academic_papers_GNAR"


def main():
    generate_vector_store()


def generate_vector_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(path_to_data, glob = "*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    
    # Defining the recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        ['\n', '.', ' ', ''],
        chunk_size=500, 
        chunk_overlap=100,
        )
    
    # Getting chunks with text splitter
    chunks = text_splitter.split_documents(documents)
    
    # Printing number of original documents and number of chunks
    print(f'Number of original document: {len(documents)} | Number of chunks; {len(chunks)}')
    
    # Printing an example of chunk content and metadata
    print("--- Example ---")
    print(f'Content; {chunks[2].page_content}')
    print(f'Metadata; {chunks[2].metadata}')

    return chunks


def save_to_chroma(chunks):
    # Delete old database
    if os.path.exists(path_to_chroma):
        shutil.rmtree(path_to_chroma)
    
    # Creating a new DB from the documents
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory = path_to_chroma)

    # Database should save automatically but we can force save using persist

    db.persist()

    print(f'Saved {len(chunks)} chunks to {path_to_chroma}')



if __name__ == "__main__":
    main()