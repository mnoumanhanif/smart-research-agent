
# Smart Summarizer & Autonomous Research Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Llama-3](https://img.shields.io/badge/Model-Llama--3--8B-blueviolet)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green)
![PEFT](https://img.shields.io/badge/Fine--Tuning-LoRA-yellow)

**An end-to-end generative AI system designed to automate the synthesis of academic literature using Multi-Agent Orchestration and Domain-Specific Fine-Tuning.**

## 🚀 Overview

This project addresses the challenge of information overload in academic research. Moving beyond standard RAG (Retrieval-Augmented Generation) pipelines, this system utilizes a **stateful multi-agent architecture** grounded by a **custom fine-tuned Large Language Model (LLM)**.

The system autonomously:
1.  **Deconstructs** user queries into optimized search terms.
2.  **Aggregates** papers from arXiv and Semantic Scholar.
3.  **Ranks** literature based on citation velocity and relevance.
4.  **Summarizes** content using a locally fine-tuned LoRA adapter.
5.  **Synthesizes** a comparative analysis report.

## 🏗️ Technical Architecture

The system is built on two core pillars:

### 1. The Engine: Fine-Tuned Llama-3 (LoRA)
* **Base Model:** Meta-Llama-3-8B.
* **Technique:** Low-Rank Adaptation (LoRA) on the arXiv summarization dataset.
* **Optimization:** 4-bit quantization (bitsandbytes) and bfloat16 precision for consumer-grade hardware inference.
* **Performance:** Achieved a **+1.3% improvement in BERTScore** over the base model with superior factual alignment.

### 2. The Brain: Multi-Agent Orchestration (LangGraph)
A Directed Acyclic Graph (DAG) of 5 specialized agents:
* **Keyword Agent:** Query decomposition and formatting.
* **Search Agent:** API interfacing (arXiv/Semantic Scholar).
* **Rank Agent:** Relevance scoring and filtering.
* **Summary Agent:** Domain-specific abstracting via LoRA adapter.
* **Compare Agent:** Cross-paper synthesis and reporting.

## 🛠️ Tech Stack

* **LLM Backbone:** Llama-3-8B (Local), Llama-3.3-70B (Groq via API).
* **Orchestration:** LangChain, LangGraph.
* **Fine-Tuning:** Hugging Face `peft`, `transformers`, `trl`.
* **Interface:** Streamlit.
* **Compute:** Optimized for NVIDIA RTX GPUs (requires CUDA).

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/smart-research-agent.git](https://github.com/yourusername/smart-research-agent.git)
   cd smart-research-agent

2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Set up API Keys**
Create a `.env` file or export variables:
```bash
export GROQ_API_KEY="your_groq_key"
export HF_TOKEN="your_huggingface_token"

```



## 🖥️ Usage

### Running the Agent Application

Launch the Streamlit interface:

```bash
streamlit run src/app.py

```

### Reproducing Training

To replicate the fine-tuning process, run the notebook in `notebooks/`:

```bash
jupyter notebook notebooks/train_lora_llama3.ipynb

```

## 📊 Evaluation

We evaluated the fine-tuned model using **LLM-as-a-Judge** and quantitative metrics (ROUGE, BERTScore).

* **Factuality Score:** 4.8/5.0
* **Coherence:** 4.7/5.0

*(See `reports/evaluation_results.pdf` for full metrics)*

## 🤝 Contributors

* **Muhammad Nouman Hanif** - AI Engineering & System Architecture
* **Syed Mujtaba Hassan** - Model Fine-Tuning & Evaluation

