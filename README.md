# SmartPark AI Parking Chatbot (Version 1)

SmartPark AI Parking Chatbot is a small **Retrieval-Augmented Generation (RAG)** project that answers questions about a parking service and allows users to create parking reservations. The system combines vector search with a language model to provide context-aware answers based on a parking knowledge base.

More updates and bug fixes are coming. A presentation will be added soon.

---

## Features

* RAG-based question answering using LangChain
* Vector search powered by Milvus
* Parking reservation storage using SQLite
* Guardrails to protect sensitive information
* Evaluation pipeline with retrieval and response metrics

---

## Project Structure

```
app.py                 # Main chatbot entry point
config.py              # Configuration (API keys, chunking, Milvus settings)
rag.py                 # RAG chain creation
milvus_store.py        # Vector store creation and loading
reservation.py         # Reservation database logic
guardrails.py          # Security and safety rules

evaluation.py          # RAG system evaluation
evaluation_dataset.json

data/
 ├─ parking_info.txt   # Knowledge base
 └─ parking_chatbot.db # SQLite reservation database
```

---

## Technologies

* Python
* LangChain
* OpenAI API
* Milvus Vector Database
* SQLite

---

## Knowledge Base

The chatbot retrieves information from `parking_info.txt`, which contains details about:

* parking prices
* working hours
* reservation policies
* EV charging
* safety rules
* additional services
* contact information

The document is split into chunks and embedded into a vector database for semantic search.

---

## Evaluation

The project includes an evaluation pipeline to measure RAG performance.

Metrics used:

* **Recall@K** – fraction of relevant information retrieved
* **Precision@K** – relevance of retrieved chunks
* **Semantic Similarity** – similarity between generated answer and reference answer
* **Accuracy** – percentage of answers above similarity threshold
* **Latency** – response time

Example results:

```
Average Recall@3: 0.769
Average Precision@3: 0.500
Average Semantic Similarity: 0.863
Accuracy: 0.692
Average Latency: 1.138 sec
```

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Set your OpenAI API key in .env file.

Run the chatbot:

```
python app.py
```

Run evaluation:

```
python evaluation.py
```

---

## Example Questions

* What are the parking prices?
* Is the parking open 24/7?
* Can electric vehicles charge here?
* How can I reserve a parking space?
* How do I cancel a reservation?

---

## Milvus details

* Milvus engine is running in docker
* Here is the instruction how to set up Milvus Standalone in docker: https://milvus.io/docs/install_standalone-docker.md