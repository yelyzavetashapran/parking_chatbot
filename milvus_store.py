# milvus_store.py
from pymilvus import connections
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from typing import List
import config


def _get_connection_args() -> dict:
    return {
        "host": config.MILVUS_HOST,
        "port": config.MILVUS_PORT
    }


def _create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def _load_documents() -> List[Document]:
    loader = TextLoader(config.PARKING_INFO_FILE)
    return loader.load()


def _split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(documents)


def _add_metadata(docs: List[Document]) -> List[Document]:
    enhanced_docs = []

    for i, doc in enumerate(docs):
        enhanced_docs.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "chunk_id": f"chunk_{i}",
                    "source": config.PARKING_INFO_FILE
                }
            )
        )

    return enhanced_docs


def start_connection():
    connections.connect(**_get_connection_args())

    print(
        f"Connected to Milvus at "
        f"{config.MILVUS_HOST}:{config.MILVUS_PORT}"
    )


def create_vector_store():

    start_connection()

    documents = _load_documents()
    split_docs = _split_documents(documents)
    enhanced_docs = _add_metadata(split_docs)

    embeddings = _create_embeddings()

    vector_store = Milvus.from_documents(
        enhanced_docs,
        embeddings,
        collection_name=config.MILVUS_COLLECTION,
        connection_args=_get_connection_args(),
    )

    print(
        f"Created {len(enhanced_docs)} chunks and stored in Milvus."
    )

    return vector_store


def load_vector_store():
    embeddings = _create_embeddings()

    return Milvus(
        embedding_function=embeddings,
        collection_name=config.MILVUS_COLLECTION,
        connection_args=_get_connection_args(),
    )