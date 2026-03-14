# Setup Guide

This guide walks you through setting up the Smart Research Agent on your local machine.

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (required for local LLM inference)
  - Minimum: 8 GB VRAM (e.g., RTX 3070)
  - Recommended: 16 GB VRAM (e.g., RTX A4000, RTX 4080)
- **CUDA Toolkit 11.8+** and compatible NVIDIA drivers
- **Git** for cloning the repository

## Step 1: Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/smart-research-agent.git
cd smart-research-agent
```

## Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `torch` with CUDA support may require a specific command for your system. See [PyTorch Get Started](https://pytorch.org/get-started/locally/).

## Step 4: Configure API Keys

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Required For | Where to Get It |
|----------|-------------|-----------------|
| `GROQ_API_KEY` | Agent app (keyword & compare agents) | [Groq Console](https://console.groq.com/keys) |
| `HF_TOKEN` | Downloading Llama-3-8B model | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `GEMINI_API_KEY` | Benchmark script only | [Google AI Studio](https://aistudio.google.com/app/apikey) |

> **Important:** You must accept the Llama 3 license on HuggingFace before downloading the model: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import langchain; print('LangChain installed successfully')"
```

## Troubleshooting

### CUDA Not Available
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Install the correct PyTorch version for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

### Out of Memory Errors
- The model uses 4-bit quantization to reduce memory usage
- Close other GPU-intensive applications
- Reduce `Papers to Analyze` slider in the app sidebar

### API Rate Limiting
- arXiv API has rate limits; the app includes early stopping to avoid throttling
- Groq free tier has request limits; consider upgrading for heavy usage
