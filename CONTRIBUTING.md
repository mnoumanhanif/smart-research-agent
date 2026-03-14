# Contributing to Smart Research Agent

Thank you for your interest in contributing! This guide will help you get started.

## How to Contribute

### Reporting Bugs

1. Check the [existing issues](https://github.com/mnoumanhanif/smart-research-agent/issues) to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include steps to reproduce, expected behavior, and your environment details.

### Suggesting Features

1. Open a new issue using the **Feature Request** template.
2. Clearly describe the feature and its use case.

### Submitting Changes

1. **Fork** the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the coding standards below.
4. Write or update tests as needed.
5. Commit with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
6. Push to your fork and open a **Pull Request**.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/smart-research-agent.git
cd smart-research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes.
- Keep functions focused and concise.
- Never commit API keys or secrets — use environment variables.

## Testing

Run the test suite before submitting:

```bash
pytest tests/
```

## Code of Conduct

Be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

## Questions?

Open an issue or reach out to the maintainers listed in the README.
