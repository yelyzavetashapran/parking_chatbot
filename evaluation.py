import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from milvus_store import load_vector_store
from rag import create_rag_chain

# METRICS FUNCTIONS

def recall_at_k(retrieved_texts, relevant_texts):
    """
    Recall@K: fraction of relevant snippets found in retrieved chunks
    """
    hits = 0
    for rel in relevant_texts:
        for chunk in retrieved_texts:
            if rel.lower() in chunk.lower():
                hits += 1
                break
    return hits / len(relevant_texts)


def precision_at_k(retrieved_texts, relevant_texts, k):
    """
    Precision@K: fraction of retrieved chunks that contain at least one relevant snippet
    """
    hits = 0
    for chunk in retrieved_texts:
        for rel in relevant_texts:
            if rel.lower() in chunk.lower():
                hits += 1
                break
    return hits / min(k, len(retrieved_texts))


def semantic_similarity_multi(answer, relevant_texts, embeddings):
    """
    Average semantic similarity of LLM answer with all relevant snippets
    """
    vec_answer = embeddings.embed_query(answer)
    scores = []
    for text in relevant_texts:
        vec_ref = embeddings.embed_query(text)
        sim = cosine_similarity([vec_answer], [vec_ref])[0][0]
        scores.append(sim)
    return sum(scores) / len(scores) if scores else 0.0

# MAIN EVALUATION FUNCTION

def evaluate(k=3):
    # Load dataset
    with open("evaluation_dataset.json", "r") as f:
        dataset = json.load(f)

    # Load vector store and retriever
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    qa_chain = create_rag_chain()
    embeddings = OpenAIEmbeddings()

    total_recall = 0
    total_precision = 0
    total_similarity = 0
    total_latency = 0
    correct_answers = 0

    print(f"\nStarting evaluation with K={k}\n")

    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        relevant_texts = sample["relevant_texts"]
        ground_truth = sample["ground_truth_answer"]

        print("=" * 60)
        print(f"Sample {i}")
        print("Question:", question)

        # ---------------- Retrieval
        docs = retriever.invoke(question)
        retrieved_texts = [doc.page_content.strip() for doc in docs]

        # Deduplicate retrieved chunks
        retrieved_texts = list(dict.fromkeys(retrieved_texts))

        recall = recall_at_k(retrieved_texts, relevant_texts)
        precision = precision_at_k(retrieved_texts, relevant_texts, k)

        total_recall += recall
        total_precision += precision

        # ---------------- LLM Response
        start_time = time.time()
        result = qa_chain.invoke({"query": question})
        response = result["result"]
        latency = time.time() - start_time
        total_latency += latency

        similarity = semantic_similarity_multi(response, relevant_texts, embeddings)
        total_similarity += similarity

        if similarity >= 0.85:
            correct_answers += 1

        # ---------------- Print details
        print("\nRetrieved Chunks:")
        for idx, chunk in enumerate(retrieved_texts, 1):
            print(f"--- Chunk {idx} ---")
            print(chunk)

        print("\nModel Answer:", response)
        print(f"Recall@{k}: {recall:.3f}")
        print(f"Precision@{k}: {precision:.3f}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Latency: {latency:.3f} sec")

    # ---------------- Final metrics
    n = len(dataset)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Average Recall@{k}: {total_recall / n:.3f}")
    print(f"Average Precision@{k}: {total_precision / n:.3f}")
    print(f"Average Semantic Similarity: {total_similarity / n:.3f}")
    print(f"Accuracy (Similarity >= 0.85): {correct_answers / n:.3f}")
    print(f"Average Latency: {total_latency / n:.3f} sec")
    print("\nEvaluation completed.\n")


if __name__ == "__main__":
    evaluate(k=3)