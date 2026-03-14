# Architecture

This document describes the technical architecture of the Smart Research Agent.

## System Overview

The Smart Research Agent is built on two core pillars:

1. **The Engine** — A fine-tuned Llama-3-8B model with LoRA adapter for domain-specific summarization
2. **The Brain** — A multi-agent orchestration pipeline using LangGraph

```
User Query
    │
    ▼
┌─────────────┐
│ Keyword Agent│  ← Groq API (Llama-3.1-8B-Instant)
└──────┬──────┘
       │ expanded keywords
       ▼
┌─────────────┐
│ Search Agent │  ← arXiv API
└──────┬──────┘
       │ raw papers
       ▼
┌─────────────┐
│  Rank Agent  │  ← Citation + recency scoring
└──────┬──────┘
       │ top-ranked papers
       ▼
┌─────────────┐
│Summary Agent │  ← Local GPU (Llama-3-8B + LoRA)
└──────┬──────┘
       │ paper summaries
       ▼
┌─────────────┐
│Compare Agent │  ← Groq API (Llama-3.3-70B)
└──────┬──────┘
       │
       ▼
  Research Report
```

## Agent Details

### 1. Keyword Agent
- **Purpose:** Expands the user's research topic into specific academic search terms
- **Model:** Llama-3.1-8B-Instant via Groq API
- **Output:** 4 comma-separated academic keywords

### 2. Search Agent
- **Purpose:** Fetches papers from academic databases
- **Source:** arXiv API
- **Behavior:** Iterates keywords, stops early when enough papers are found (max 5)
- **Deduplication:** Removes duplicate papers based on title

### 3. Rank Agent
- **Purpose:** Scores and selects the most relevant papers
- **Scoring Formula:** `40% citation score + 60% recency score`
- **Selection:** Returns top N papers (configurable via UI slider)

### 4. Summary Agent
- **Purpose:** Generates concise summaries of each paper's abstract
- **Model:** Meta-Llama-3-8B with LoRA adapter (`Mujtaba007/llama3-arxiv-lora`)
- **Quantization:** 4-bit (NF4) via bitsandbytes for efficient GPU inference
- **Fallback:** Uses base model if adapter loading fails

### 5. Compare Agent
- **Purpose:** Synthesizes a structured research report from all summaries
- **Model:** Llama-3.3-70B-Versatile via Groq API
- **Output Sections:** Executive Summary, Key Themes, Critical Analysis, Future Directions

## State Management

The pipeline uses LangGraph's `StateGraph` with a `TypedDict` state:

```python
class ResearchState(TypedDict):
    user_query: str
    expanded_keywords: List[str]
    raw_papers: List[Dict[str, Any]]
    selected_papers: List[Dict[str, Any]]
    summaries: Dict[str, str]
    final_analysis: str
    logs: List[str]
```

Each agent reads from and writes to this shared state, enabling a clean data flow through the pipeline.

## Model Fine-Tuning (Phase 1)

The LoRA adapter was trained on the `ccdv/arxiv-summarization` dataset:

- **Base Model:** Meta-Llama-3-8B
- **Technique:** Low-Rank Adaptation (LoRA)
- **Quantization:** 4-bit (QLoRA) for training efficiency
- **Result:** +1.3% BERTScore improvement over the base model
- **Adapter:** Hosted at `Mujtaba007/llama3-arxiv-lora` on HuggingFace Hub

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM (Local) | Meta-Llama-3-8B + LoRA |
| LLM (Cloud) | Llama-3.1-8B, Llama-3.3-70B via Groq |
| Orchestration | LangGraph, LangChain |
| Fine-Tuning | HuggingFace PEFT, Transformers |
| Web UI | Streamlit |
| Paper Search | arXiv API |
| Quantization | bitsandbytes (4-bit NF4) |
| Evaluation | ROUGE, BERTScore, Gemini LLM-as-Judge |
