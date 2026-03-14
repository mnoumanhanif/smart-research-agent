# Smart Research Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Llama-3](https://img.shields.io/badge/Model-Llama--3--8B-blueviolet)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green)
![PEFT](https://img.shields.io/badge/Fine--Tuning-LoRA-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end generative AI system that automates the synthesis of academic literature using **multi-agent orchestration** and **domain-specific fine-tuning**.

## Overview

This project addresses the challenge of information overload in academic research. Moving beyond standard RAG (Retrieval-Augmented Generation) pipelines, the system uses a **stateful multi-agent architecture** powered by a **custom fine-tuned LLM**.

The system autonomously:

1. **Deconstructs** user queries into optimized academic search terms
2. **Aggregates** papers from arXiv
3. **Ranks** literature based on citation velocity and relevance
4. **Summarizes** content using a locally fine-tuned LoRA adapter
5. **Synthesizes** a comparative analysis report

## Key Features

- **5-Agent Pipeline** — Keyword expansion → Paper search → Ranking → Summarization → Comparative analysis
- **Fine-Tuned LLM** — Llama-3-8B with LoRA adapter achieving +1.3% BERTScore improvement
- **4-Bit Quantization** — Runs on consumer GPUs (8 GB+ VRAM) via bitsandbytes
- **Interactive UI** — Streamlit web interface with configurable parameters
- **Streaming Execution** — Real-time progress updates as each agent completes

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (Local) | Meta-Llama-3-8B + LoRA adapter |
| LLM (Cloud) | Llama-3.1-8B, Llama-3.3-70B via Groq |
| Orchestration | LangGraph, LangChain |
| Fine-Tuning | HuggingFace PEFT, Transformers |
| Web UI | Streamlit |
| Paper Search | arXiv API |
| Quantization | bitsandbytes (4-bit NF4) |
| Evaluation | ROUGE, BERTScore, Gemini LLM-as-Judge |

## Project Structure

```
smart-research-agent/
├── src/
│   ├── agent_app.py               # Main Streamlit application
│   └── benchmark.py               # Model evaluation script
├── notebook/
│   └── Phase1_Training_Notebook_(24K-8001).ipynb
├── reports/
│   └── Gen_AI_Assignment_04__(24K_8001).pdf
├── docs/                          # Documentation
│   ├── setup.md                   # Detailed setup guide
│   ├── architecture.md            # System architecture
│   └── development.md             # Development guide
├── tests/                         # Test suite
│   └── test_agents.py
├── .github/                       # CI/CD and templates
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
├── CONTRIBUTING.md                # Contribution guidelines
├── CHANGELOG.md                   # Version history
├── LICENSE                        # MIT License
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (8 GB+ VRAM)
- [Groq API key](https://console.groq.com/keys)
- [HuggingFace token](https://huggingface.co/settings/tokens)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/smart-research-agent.git
cd smart-research-agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

> For detailed setup instructions including GPU configuration, see [docs/setup.md](docs/setup.md).

## Usage

### Running the Agent Application

```bash
export GROQ_API_KEY="your_groq_key"
export HF_TOKEN="your_huggingface_token"

streamlit run src/agent_app.py
```

The application opens in your browser. Enter a research topic and click **Start Research** to begin the automated literature review.

### Running the Benchmark

```bash
export HF_TOKEN="your_huggingface_token"
export GEMINI_API_KEY="your_gemini_key"

python src/benchmark.py
```

### Reproducing the Training

```bash
jupyter notebook notebook/Phase1_Training_Notebook_(24K-8001).ipynb
```

## Evaluation

The fine-tuned model was evaluated using **LLM-as-a-Judge** and quantitative metrics:

| Metric | Score |
|--------|-------|
| Factuality | 4.8 / 5.0 |
| Coherence | 4.7 / 5.0 |
| BERTScore | +1.3% over base model |

See `reports/Gen_AI_Assignment_04__(24K_8001).pdf` for the full evaluation report.

## Development

See the [Development Guide](docs/development.md) for information on:
- Code conventions and project structure
- Adding new agents to the pipeline
- Running tests

### Testing

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs and requesting features
- Submitting pull requests
- Code style and conventions

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Muhammad Nouman Hanif** — AI Engineering & System Architecture
- **Syed Mujtaba Hassan** — Model Fine-Tuning & Evaluation
