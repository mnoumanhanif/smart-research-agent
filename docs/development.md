# Development Guide

This guide covers development workflows and conventions for the Smart Research Agent.

## Project Structure

```
smart-research-agent/
├── src/                           # Source code
│   ├── agent_app.py               # Main Streamlit agent application
│   └── benchmark.py               # Model evaluation/benchmark script
├── notebook/                      # Jupyter notebooks
│   └── Phase1_Training_Notebook_(24K-8001).ipynb
├── reports/                       # Generated reports and evaluations
│   └── Gen_AI_Assignment_04__(24K_8001).pdf
├── docs/                          # Documentation
│   ├── setup.md
│   ├── architecture.md
│   └── development.md
├── tests/                         # Test suite
│   └── test_agents.py
├── .github/                       # GitHub configuration
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── .gitignore
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

## Running the Application

### Streamlit Agent App

```bash
# Set environment variables
export GROQ_API_KEY="your_key"
export HF_TOKEN="your_token"

# Launch the app
streamlit run src/agent_app.py
```

### Benchmark Script

```bash
# Requires GPU and all API keys
export HF_TOKEN="your_token"
export GEMINI_API_KEY="your_key"

python src/benchmark.py
```

### Training Notebook

```bash
jupyter notebook notebook/Phase1_Training_Notebook_(24K-8001).ipynb
```

## Code Conventions

### Python Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings to all public functions

### Environment Variables
- Never hardcode API keys or secrets
- Use `os.environ.get()` with sensible defaults
- Document required variables in `.env.example`

### Error Handling
- Use specific exception types (not bare `except`)
- Provide meaningful error messages
- Include fallback behavior where appropriate

## Adding a New Agent

1. Define the agent function in `src/agent_app.py`:
   ```python
   def my_new_agent(state: ResearchState):
       """Description of what this agent does."""
       logs = state.get("logs", [])
       # Agent logic here
       logs.append("✅ My Agent Complete")
       return {"new_field": result, "logs": logs}
   ```

2. Add the field to `ResearchState` if needed:
   ```python
   class ResearchState(TypedDict):
       # ... existing fields
       new_field: str
   ```

3. Wire it into the graph:
   ```python
   workflow.add_node("my_agent", my_new_agent)
   workflow.add_edge("previous_agent", "my_agent")
   workflow.add_edge("my_agent", "next_agent")
   ```

## Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```
