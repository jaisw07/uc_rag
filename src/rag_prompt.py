import ollama
from typing import List


# -----------------------------
# Prompt Builder
# -----------------------------
def build_rag_prompt(context_chunks: List[str], question: str) -> str:
    """
    Build a structured RAG prompt

    Args:
        context_chunks: List of retrieved text chunks
        question: User query

    Returns:
        Formatted prompt string
    """

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert AI assistant for answering questions based on documents.

Instructions:
- Answer ONLY using the provided context
- Do NOT make up information
- If the answer is not in the context, say "I don't know"
- Keep answers concise and accurate
- Refer to the metadata to filter relevant course-specific information and cite sources for the information you provide

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


# -----------------------------
# LLM Call Wrapper
# -----------------------------
def generate_answer(
    model_name: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """
    Generate response from Ollama model

    Args:
        model_name: llama3.1:8b | gemma2:9b | phi3:mini
        prompt: Final prompt string
        temperature: Controls randomness
        max_tokens: Output length

    Returns:
        Model response text
    """

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    )

    return response["message"]["content"]


# -----------------------------
# Full RAG Pipeline Step
# -----------------------------
def rag_inference(
    model_name: str,
    retrieved_chunks: List[str],
    query: str
):
    """
    End-to-end RAG step

    Args:
        model_name: LLM to use
        retrieved_chunks: Top-k retrieved chunks
        query: User question

    Returns:
        Answer string
    """

    prompt = build_rag_prompt(retrieved_chunks, query)

    answer = generate_answer(
        model_name=model_name,
        prompt=prompt
    )

    return answer