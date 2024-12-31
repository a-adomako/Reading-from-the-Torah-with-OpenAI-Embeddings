# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']
print(f"OpenAI API Key Loaded: {openai.api_key is not None}")

CHROMA_PATH = "chroma"
DATA_PATH = "path/to/data"

def main():
    print("Starting the process...")
    generate_data_store()

def generate_data_store():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")
    chunks = split_text(documents)
    print(f"Generated {len(chunks)} chunks.")


def generate_data_store():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")
    chunks = split_text(documents)
    print(f"Generated {len(chunks)} chunks.")


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    if not documents:
        print(f"No documents were loaded from {DATA_PATH}. Please check the data path and file format.")
    else:
        print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    if not documents:
        print("No documents provided to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        print(f"Example chunk: {chunks[0].page_content[:00]}...")
    else:
        print("No chunks were generated.")
    return chunks


if __name__ == "__main__":
    main()
