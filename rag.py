# rag.py
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


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )


def _create_prompt() -> PromptTemplate:
    prompt_template = """
    Use ONLY the information from the provided context.

    If the answer is not present in the context, say:
    "I don't have that information."

    Do not invent rules, policies, or details.

    Context:
    {context}

    Question: {question}

    Answer clearly and briefly.
    """ 

    return PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n" + prompt_template
    )


def _create_retriever():
    vector_store = load_vector_store()
    return vector_store.as_retriever()


def create_rag_chain():
    retriever = _create_retriever()
    llm = _create_llm()
    prompt = _create_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain