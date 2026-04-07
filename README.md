# 📚 RAG Pipeline + LLM Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&style=flat-square)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red?style=flat-square)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-SBERT-purple?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Overview

This project implements a **modular Retrieval-Augmented Generation (RAG) pipeline** for domain-specific question answering, along with a **comprehensive evaluation framework** for:

- 🔎 Retrieval performance  
- 🧠 LLM generation quality  
- ⚖️ Comparison across models, embeddings, and vector databases  

It is designed as part of a **Generative AI coursework project**, covering:

- **Assignment 1** → RAG system design
- **Assignment 2** → LLM evaluation & experimentation

---

## 🧠 Key Features

### 🔹 RAG Pipeline

- Document ingestion & preprocessing  
- Configurable chunking strategies  
- Multiple embedding models (MiniLM, BGE, etc.)  
- Dual vector DB support:
  - **ChromaDB**
  - **Qdrant**
- Semantic retrieval + ranking  
- Prompt construction with context injection  
- LLM-based answer generation (via Ollama)  

### 🔹 Evaluation Framework

- **Retrieval metrics:**
  - Top-1 Accuracy  
  - Hit Rate @ K  
- **Generation metrics:**
  - F1 Score  
  - Faithfulness  
  - Relevancy  
- **Multi-LLM comparison:**
  - LLaMA variants  
  - Gemma  
  - Phi models  

---

## 📂 Project Structure

```
src/
├── chunking.py
├── embedding_chromadb.py
├── embedding_qdrant.py
├── ingestion_and_preprocessing.py
├── query.py
├── rag_prompt.py
├── retrieval_evaluation.py
├── llm_evaluation.py
├── README.md
├── environment.yml
└── orchestrator.ipynb
```

---

## ⚙️ Installation

### 1️⃣ Create Environment

```bash
conda env create -f environment.yml
conda activate rag-env
```

### 2️⃣ Start Qdrant (Vector DB)

```bash
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### 3️⃣ Install & Run Ollama Models

```bash
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull phi3:mini
```

---

## 🔄 Pipeline Workflow

```
Raw Documents
     ↓
Ingestion & Cleaning
     ↓
Chunking
     ↓
Embeddings
     ↓
Vector DB (ChromaDB / Qdrant)
     ↓
Query → Retrieval
     ↓
Prompt Injection
     ↓
LLM Response
```

---

## 🧪 Experiments Conducted

### 🔹 Retrieval Experiments

| Aspect | Details |
|--------|---------|
| **Embeddings Compared** | MiniLM (384d), BGE (512d) |
| **Vector Databases** | ChromaDB vs Qdrant |
| **Metrics** | Top-1 Accuracy, Hit@K |

### 🔹 LLM Evaluation

| Aspect | Details |
|--------|---------|
| **Models Evaluated** | `llama3.1:8b`, `gemma2:9b`, `phi3:mini` |
| **Metrics** | F1 Score, Faithfulness, Relevancy |

---

## 📊 Example Results

| Model | F1 Score | Faithfulness | Relevancy |
|-------|----------|--------------|-----------|
| Phi3 Mini | 0.41 | 0.95 | 0.94 |
| Gemma 2 9B | 0.53 | 0.96 | 0.96 |
| LLaMA 3.1 | 0.42 | 0.95 | 0.94 |

---

## ▶️ Usage

### Run Full Pipeline

```bash
jupyter notebook orchestrator.ipynb
```

### Run Retrieval Evaluation

```python
from src.retrieval_evaluation import *

generate_eval_dataset(...)
evaluate_retrieval(...)
```

### Run LLM Evaluation

```python
from src.llm_evaluation import *

evaluate_llms(...)
```

---

## 🧩 Design Highlights

- ✅ Fully modular architecture  
- ✅ Swappable embedding models  
- ✅ Multi-vector DB support  
- ✅ Plug-and-play LLM evaluation  
- ✅ Optimized for low VRAM GPUs (RTX 4050)  

---

## 👨‍💻 Author

**Shrey Jaiswal**
