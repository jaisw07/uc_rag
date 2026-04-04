import json
from tqdm import tqdm
import ollama

from src.rag_prompt import rag_inference
from src.query import query_chromadb, query_qdrant


# =========================
# 1. F1 SCORE
# =========================
def compute_f1(pred, truth):
    pred_tokens = set(pred.lower().split())
    truth_tokens = set(truth.lower().split())

    common = pred_tokens & truth_tokens

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)


# =========================
# 2. LLM JUDGE CALL
# =========================
def llm_judge(prompt: str, model="mistral:7b-instruct"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        output = response["message"]["content"].strip()

        # extract numeric score
        score = float(output)
        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"⚠️ LLM judge error: {e}")
        return 0.0


# =========================
# 3. FAITHFULNESS
# =========================
def evaluate_faithfulness(question, context, answer):
    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context?

Score from 0 to 1:
- 1 = fully grounded in context
- 0 = hallucinated or unsupported

Only output a number.
"""
    return llm_judge(prompt)


# =========================
# 4. RELEVANCY
# =========================
def evaluate_relevancy(question, answer):
    prompt = f"""
Question:
{question}

Answer:
{answer}

How relevant is the answer to the question?

Score from 0 to 1:
- 1 = perfectly answers
- 0 = irrelevant

Only output a number.
"""
    return llm_judge(prompt)


# =========================
# 5. MAIN EVALUATION FUNCTION
# =========================
def evaluate_llms(
    eval_dataset_path: str,
    retrieval_fn,
    embedding_model: str,
    collection_name: str,
    llm_models: list,
    top_k: int = 5,
    max_samples: int = None,
):
    """
    Evaluate multiple LLMs on same retrieval pipeline

    Args:
        eval_dataset_path: path to eval dataset
        retrieval_fn: query_chromadb or query_qdrant
        embedding_model: embedding used for retrieval
        collection_name: DB collection
        llm_models: list of models to evaluate
        top_k: number of retrieved chunks
    """

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]

    results = {}

    # =========================
    # LOOP OVER MODELS
    # =========================
    for model_name in llm_models:
        print(f"\n🚀 Evaluating: {model_name}")

        f1_scores = []
        faith_scores = []
        rel_scores = []

        for sample in tqdm(dataset):
            question = sample["question"]
            ground_truth = sample["ground_truth_answer"]
            source_file = sample["source_file"]

            # =========================
            # RETRIEVE
            # =========================
            retrieved_chunks = retrieval_fn(
                query=question,
                model_name=embedding_model,
                collection_name=collection_name,
                top_k=top_k,
                source_file=f"{source_file}.json",
            )

            context = "\n\n".join(retrieved_chunks)

            # =========================
            # GENERATE
            # =========================
            answer = rag_inference(
                model_name=model_name,
                retrieved_chunks=retrieved_chunks,
                query=question,
            )

            # =========================
            # METRICS
            # =========================
            f1 = compute_f1(answer, ground_truth)
            faith = evaluate_faithfulness(question, context, answer)
            rel = evaluate_relevancy(question, answer)

            f1_scores.append(f1)
            faith_scores.append(faith)
            rel_scores.append(rel)

        # =========================
        # AGGREGATE
        # =========================
        results[model_name] = {
            "f1": sum(f1_scores) / len(f1_scores),
            "faithfulness": sum(faith_scores) / len(faith_scores),
            "relevancy": sum(rel_scores) / len(rel_scores),
        }

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n📊 LLM Evaluation Results:")
    for model, metrics in results.items():
        print(f"\n🔹 {model}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return results