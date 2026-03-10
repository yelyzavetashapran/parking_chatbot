# SmartPark AI Parking Chatbot (Version 2)

SmartPark AI Parking Chatbot is a small **Retrieval-Augmented Generation (RAG)** project that answers questions about a parking service and allows users to create parking reservations. The system combines vector search with a language model to provide context-aware answers based on a parking knowledge base.

More updates and bug fixes are coming.

---

## Features

* RAG-based question answering using LangChain
* Vector search powered by Milvus
* Parking reservation storage using SQLite
* Guardrails to protect sensitive information
* Evaluation pipeline with retrieval and response metrics
* Human-in-the-loop for reservation confirmation/refusal
* Email notification after review of the request by the administrator

---

## Project Structure

```
app.py                 # Main chatbot entry point
config.py              # Configuration (API keys, chunking, Milvus settings)
rag.py                 # RAG chain creation
milvus_store.py        # Vector store creation and loading
reservation.py         # Reservation database logic
reservation_graph.py   # LangGraph implementation 
email_service.py       # Email notification
admin_api.py           # API implementation
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
* LangGraph
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

Set your OPENAI_API_KEY, EMAIL_USER, EMAIL_PASSWORD in .env file.
Set up Milvus Standalone (https://milvus.io/docs/install_standalone-docker.md)

Run API command in separate terminal:

```
uvicorn admin_api:app --reload
```

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


## Example of usage

Run docker container with Milvus Standalone:
![alt text](example_usage_screenshots\image-1.png)

Run API service:
![alt text](example_usage_screenshots\image.png)

API admin service is running:

![alt text](example_usage_screenshots\image-2.png)
![alt text](example_usage_screenshots\image-3.png)

Run app.py in separated terminal and start use chatbot:
![alt text](example_usage_screenshots\image-4.png)

Questions answers:

![alt text](example_usage_screenshots\image-5.png)

Reservation flow:

1) collect user's data and send it to admin review

![alt text](example_usage_screenshots\image-6.png)

2) ask bot about reservation status before admin approve

![alt text](example_usage_screenshots\image-7.png)

3) check API and approve, check if email is sent, and also reask chatbot about reservation status

![alt text](example_usage_screenshots\image-8.png)
![alt text](example_usage_screenshots\image-9.png)
![alt text](example_usage_screenshots\image-10.png)
![alt text](example_usage_screenshots\image-11.png)

4) type 'exit' to finish chat

![alt text](example_usage_screenshots\image-12.png)