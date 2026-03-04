# milvus_store.py

from pymilvus import connections
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import config


def start_connection():
    connections.connect(
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")


def create_vector_store():
    """
    Creates vector store from parking_info.txt
    Applies improved chunking and metadata.
    """
    start_connection()

    loader = TextLoader(config.PARKING_INFO_FILE)
    documents = loader.load()

    # Better chunking for small dataset
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,        # recommend: 300
        chunk_overlap=config.CHUNK_OVERLAP,  # recommend: 30
        separators=["\n\n", "\n", ".", " "]
    )

    split_docs = splitter.split_documents(documents)

    # Add metadata (chunk_id + source)
    enhanced_docs = []
    for i, doc in enumerate(split_docs):
        enhanced_docs.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "chunk_id": f"chunk_{i}",
                    "source": config.PARKING_INFO_FILE
                }
            )
        )

    embeddings = OpenAIEmbeddings()

    vector_store = Milvus.from_documents(
        enhanced_docs,
        embeddings,
        collection_name=config.MILVUS_COLLECTION,
        connection_args={
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT
        },
    )

    print(f"Created {len(enhanced_docs)} chunks and stored in Milvus.")

    return vector_store


def load_vector_store():
    """
    Loads existing Milvus collection.
    """
    embeddings = OpenAIEmbeddings()

    return Milvus(
        embedding_function=embeddings,
        collection_name=config.MILVUS_COLLECTION,
        connection_args={
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT
        },
    )