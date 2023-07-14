#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
from chromadb.config import Settings

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


load_dotenv()


# Load environment variables
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")



def process_documents(doc_path) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading document: {doc_path}")
    documents = load_single_document(doc_path)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {doc_path}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    

    files = os.listdir(source_directory)
    if len(files) == 0:
        print("No files found in source directory")
        exit(0)
    else:
        if not os.path.exists("db"):
            os.makedirs("db")
        for file in files:
            folder = file.split(".")[0]
            if not os.path.exists(f"db/{folder}"):
                os.makedirs(f"db/{folder}")
                print("Creating new vectorstore")
                texts = process_documents(f"{source_directory}/{file}")
                print(f"Creating embeddings. May take some minutes...")
                CHROMA_SETTINGS = Settings(
                    chroma_db_impl='duckdb+parquet',
                    persist_directory = 'db/'+folder,
                    anonymized_telemetry=False
                )
            
                db = Chroma.from_documents(texts, embeddings, persist_directory="db/"+folder, client_settings=CHROMA_SETTINGS)
            else:
                if os.path.exists(os.path.join("db/"+folder, 'index')):
                    if os.path.exists(os.path.join("db/"+folder, 'chroma-collections.parquet')) and os.path.exists(os.path.join("db/"+folder, 'chroma-embeddings.parquet')):
                        list_index_files = glob.glob(os.path.join("db/"+folder, 'index/*.bin'))
                        list_index_files += glob.glob(os.path.join("db/"+folder, 'index/*.pkl'))
                        # At least 3 documents are needed in a working vectorstore
                        if len(list_index_files) > 3:
                            print("Vectorstore already exists. Skipping ingestion")


    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
