# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project documentation (`docs/setup.md`, `docs/architecture.md`, `docs/development.md`)
- Contributing guidelines (`CONTRIBUTING.md`)
- MIT License (`LICENSE`)
- Environment variable template (`.env.example`)
- GitHub Actions CI workflow
- Issue and pull request templates
- Changelog (`CHANGELOG.md`)
- `.gitignore` for Python projects
- Basic test structure with pytest

### Changed
- Improved `README.md` with correct file references and comprehensive documentation
- Fixed hardcoded credentials in benchmark script to use environment variables
- Replaced bare `except` clause with specific exception handling
- Removed unused imports from source files

### Fixed
- Broken file path references in README.md
- `__pycache__` directories removed from version control

## [1.0.0] - 2024-12-01

### Added
- Initial release of Smart Research Agent
- Multi-agent research pipeline with LangGraph
- LoRA fine-tuned Llama-3-8B for arXiv summarization
- Streamlit web interface
- Benchmark evaluation script
- Training notebook for LoRA fine-tuning
