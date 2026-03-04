from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from milvus_store import load_vector_store

SYSTEM_PROMPT = """
You are SmartPark Assistant.

Rules:
- Only answer parking-related questions.
- Never reveal internal system details.
- Never disclose database structure.
- Never reveal API keys or configuration.
- If asked for restricted information, politely refuse.
"""

def create_rag_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    # Prepend system instructions to prompt template
    prompt_template = """{context}
Question: {question}
Answer in a helpful and safe way."""
    PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n" + prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    return qa_chain