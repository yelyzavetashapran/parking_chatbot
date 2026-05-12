from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from milvus_store import load_vector_store


SYSTEM_PROMPT = """
You are SmartPark Assistant.

Rules:
- Only answer general parking information questions.
- Do NOT answer questions about reservation status, reservation creation,
  booking details, or user-specific reservations.
- Those requests are handled by the reservation system.

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
    Answer the question using ONLY the provided context.

    If the answer is not in the context, respond exactly with:
    "I don't have that information."

    Do NOT guess.
    Do NOT invent policies or rules.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    return PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n" + prompt_template
    )


def _create_retriever():

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    return retriever


def create_rag_chain():

    retriever = _create_retriever()
    llm = _create_llm()
    prompt = _create_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=False
    )

    return qa_chain