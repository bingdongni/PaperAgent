# PaperAgent: Academic Multi-Agent Collaboration Framework

<div align="center">

![PaperAgent Logo](https://img.shields.io/badge/PaperAgent-v1.0.0-blue)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/yourusername/paperagent?style=social)

**Your AI-Powered Research Assistant for the Complete Academic Lifecycle**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-demo) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

**PaperAgent** is a production-grade, academic-level multi-agent collaboration framework designed for graduate students, researchers, and academic teams. It covers the entire research workflow:

**Research Topic Selection → Literature Review → Experiment Design → Paper Writing → Formatting → Submission → Results Management**

Built on the **KtR (Knowledge-to-Role)** framework, PaperAgent systematically decomposes research tasks to avoid common multi-agent pitfalls like coordination overhead, performance redundancy, and debugging difficulties.

### 🎯 Core Goals

- ✅ **Local Privacy Protection**: All data stays on your machine
- ✅ **Compliant Data Collection**: Respects API terms of service (arXiv, Google Scholar)
- ✅ **Academic Rigor**: Multi-layer quality control for publication standards
- ✅ **Efficient Collaboration**: KtR-driven agent coordination

---

## ✨ Features

### 🔍 Literature Research (Literature Agent)
- **Smart Topic Recommendation**: AI-powered research topic suggestions based on field analysis
- **Multi-source Paper Collection**: Integrated arXiv & Google Scholar search
- **Deep Paper Analysis**: Automatic summarization, clustering, and citation tracking
- **Research Gap Identification**: Discover underexplored areas in your field
- **Citation Management**: BibTeX generation and formatting

### 🔬 Experiment Management (Experiment Agent)
- **Experiment Design**: AI-guided experimental methodology
- **Data Analysis**: Statistical testing and visualization recommendations
- **Results Visualization**: Automatic figure and table generation
- **Experiment-Paper Sync**: Seamless integration with paper writing

### ✍️ Academic Writing (Writing Agent)
- **Structured Drafting**: Generate complete paper outlines
- **Section Writing**: AI-assisted writing for all paper sections
- **Academic Polishing**: Grammar, style, and clarity improvements
- **Multi-format Citations**: Support for IEEE, APA, MLA, GB/T 7714, etc.
- **Bilingual Support**: English and Chinese academic writing

### 📝 Formatting & Submission (Formatting Agent)
- **Journal Templates**: 50+ pre-built templates (IEEE, Elsevier, Springer, etc.)
- **LaTeX Integration**: Full LaTeX support with Overleaf sync
- **Format Conversion**: Word ↔ LaTeX ↔ PDF
- **Submission Checklist**: Automated compliance checking

### 🎯 Orchestration (Boss Agent)
- **Task Decomposition**: Intelligent workflow planning
- **Progress Monitoring**: Real-time project tracking
- **Quality Control**: Multi-layer validation and review
- **Error Recovery**: Automatic retry and fallback mechanisms

### 🛠️ Additional Features
- **Local LLM Support**: Ollama integration (Llama 3, Qwen2, etc.)
- **Cloud LLM Support**: OpenAI GPT-4, Anthropic Claude
- **Docker Deployment**: One-command setup
- **Web Interface**: Intuitive Streamlit UI
- **REST API**: Full-featured FastAPI backend

### 🎨 **NEW** - Enhanced Capabilities

#### 📊 Advanced Statistical Analysis

- **Deep Statistical Methods**: scipy, statsmodels, scikit-learn integration
- **Comprehensive Tests**: t-tests, ANOVA, regression, chi-square, Mann-Whitney U
- **Machine Learning**: Random Forest, cross-validation, feature importance
- **Effect Size Calculation**: Cohen's d, eta-squared, Cramér's V

#### 🎓 Advanced LaTeX Formatting

- **Complex Layouts**: Multi-column layouts, custom environments
- **Algorithm Formatting**: Professional algorithm and pseudocode environments
- **Theorem Environments**: Theorems, lemmas, proofs, definitions with automatic numbering
- **Advanced Tables**: Booktabs, multi-row/column, long tables spanning pages
- **Mathematical Formatting**: Matrices, aligned equations, cases, integrals, derivatives

#### 🖼️ Multimodal Analysis

- **Chart Understanding**: Automatic chart type detection (bar, line, pie, scatter, histogram)
- **Data Extraction**: Extract data points and trends from chart images
- **Formula Recognition**: OCR-based mathematical formula recognition with LaTeX conversion
- **Table Analysis**: Table structure detection and data extraction from images
- **Document Structure**: Section extraction, citation parsing, reference analysis
- **Code Analysis**: Complexity metrics, quality assessment, structure analysis

#### 📈 Publication-Quality Visualizations

- **Static Plots**: Scatter, bar, line, heatmap, box, violin, histogram plots
- **Interactive Visualizations**: Plotly-based interactive charts and 3D plots
- **Correlation Matrices**: Beautiful correlation heatmaps with statistical significance
- **Academic Style**: Publication-ready figures with 300 DPI, color-blind friendly palettes

#### 📄 Deep PDF Analysis

- **Full Text Extraction**: Complete text extraction with NLP analysis
- **Table Extraction**: Automatic table detection and conversion to DataFrames
- **Image Analysis**: Extract and analyze all images and figures
- **Structure Analysis**: Document hierarchy, sections, and cross-references
- **Metadata Extraction**: Author, title, keywords, abstract parsing

#### 🔗 Integrated Workflow

- **One-Stop Analysis**: Unified interface for all multimodal content
- **Auto-Detection**: Automatically detect content type (PDF, image, code, text)
- **Seamless Integration**: All tools work together seamlessly

See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for detailed usage guide and examples.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional but recommended)
- PostgreSQL 16+ (or use Docker)
- Redis 7+ (or use Docker)

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/paperagent.git
cd paperagent

# Copy environment file
cp .env.example .env

# Edit .env with your API keys (optional, can use local LLM)
nano .env

# Start all services
docker-compose up -d

# Access the web interface
# http://localhost:8501

# Access the API documentation
# http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/paperagent.git
cd paperagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env file with your configuration

# Initialize database
python -c "from paperagent.database import init_db; init_db()"

# Start API server
uvicorn paperagent.api.main:app --host 0.0.0.0 --port 8000

# In another terminal, start web interface
streamlit run paperagent/web/app.py
```

### Option 3: Local LLM Setup (Privacy-Focused)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Update .env
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3

# Start PaperAgent
docker-compose up -d
```

---

## 📚 Usage Examples

### Example 1: Complete Research Workflow

```python
from paperagent.agents import BossAgent

# Initialize boss agent
boss = BossAgent()

# Create project
project = boss.execute({
    'action': 'create_project',
    'name': 'Deep Learning for Medical Imaging',
    'research_field': 'Computer Science',
    'description': 'Investigating CNN architectures for medical image diagnosis',
    'keywords': ['deep learning', 'medical imaging', 'CNN']
})

# Execute full workflow
result = boss.execute({
    'action': 'execute_workflow',
    'project_id': project['project_id']
})

# Check progress
progress = boss.execute({
    'action': 'monitor_progress',
    'project_id': project['project_id']
})

print(f"Progress: {progress['progress_percentage']}%")
```

### Example 2: Literature Search Only

```python
from paperagent.agents import LiteratureAgent

lit_agent = LiteratureAgent()

# Search papers
result = lit_agent.execute({
    'action': 'search_literature',
    'query': 'transformer models for NLP',
    'max_results': 50,
    'sources': ['arxiv', 'google_scholar'],
    'project_id': 1
})

print(f"Found {result['total_papers']} papers")

# Identify research gaps
gaps = lit_agent.execute({
    'action': 'identify_gaps',
    'project_id': 1,
    'research_field': 'Natural Language Processing'
})

for gap in gaps['research_gaps']:
    print(f"Gap: {gap['gap']}")
    print(f"Importance: {gap['importance']}")
```

### Example 3: Paper Writing

```python
from paperagent.agents import WritingAgent

writer = WritingAgent()

# Create paper structure
structure = writer.execute({
    'action': 'create_structure',
    'title': 'Advances in Transformer Architectures',
    'objective': 'Survey recent transformer improvements',
    'project_id': 1,
    'findings': ['Efficient attention mechanisms', 'Sparse transformers']
})

# Write introduction
intro = writer.execute({
    'action': 'write_section',
    'section': 'introduction',
    'context': 'Survey paper on transformers',
    'paper_id': structure['paper_id'],
    'key_points': ['Background', 'Motivation', 'Contributions']
})

# Generate abstract
abstract = writer.execute({
    'action': 'write_abstract',
    'paper_id': structure['paper_id'],
    'word_limit': 250
})
```

---

## 🏗️ Architecture

```
PaperAgent/
├── paperagent/
│   ├── agents/              # AI Agents
│   │   ├── boss_agent.py         # Central orchestrator
│   │   ├── literature_agent.py   # Literature research
│   │   ├── experiment_agent.py   # Experiment management
│   │   ├── writing_agent.py      # Paper writing
│   │   └── base_agent.py         # Base agent class
│   ├── api/                 # FastAPI backend
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routers/
│   ├── web/                 # Streamlit frontend
│   │   └── app.py
│   ├── core/                # Core utilities
│   │   ├── config.py
│   │   ├── llm_manager.py
│   │   └── prompts.py
│   ├── database/            # Database models
│   │   ├── models.py
│   │   └── database.py
│   └── tools/               # Tool modules
│       ├── literature_collector.py
│       ├── latex_processor.py
│       └── document_processor.py
├── data/                    # Data storage
│   ├── papers/
│   ├── experiments/
│   └── literature/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://paperagent:password@localhost:5432/paperagent

# LLM Provider (openai, anthropic, ollama)
DEFAULT_LLM_PROVIDER=ollama

# OpenAI (optional)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o

# Anthropic (optional)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Literature Collection
USE_PROXY=false  # Enable for Google Scholar
ARXIV_MAX_RESULTS=50
```

---

## 🆚 Comparison

### vs. Traditional Tools

| Feature | Zotero+Word+Grammarly | PaperAgent |
|---------|----------------------|------------|
| Full Workflow | ❌ Fragmented | ✅ Integrated |
| AI Assistance | ❌ Limited | ✅ End-to-end |
| Local Privacy | ⚠️ Cloud-based | ✅ Local-first |
| Multi-agent | ❌ No | ✅ KtR-driven |

### vs. General Multi-Agent Frameworks

| Feature | AutoGen/CrewAI | PaperAgent |
|---------|----------------|------------|
| Domain-specific | ❌ Generic | ✅ Academic-focused |
| Agent Design | ⚠️ Manual | ✅ KtR-optimized |
| Academic Tools | ❌ Limited | ✅ Comprehensive |
| Quality Control | ❌ Basic | ✅ Multi-layer |

---

## 🛣️ Roadmap

- [x] Core agent framework
- [x] Literature collection and analysis
- [x] Experiment design support
- [x] Paper writing and polishing
- [x] Web interface
- [x] REST API
- [ ] VS Code extension
- [ ] Batch processing
- [ ] Team collaboration features
- [ ] Custom journal templates marketplace
- [ ] Multilingual support (more languages)
- [ ] Integration with reference managers (Mendeley, EndNote)
- [ ] Plagiarism checking
- [ ] Submission tracking

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/paperagent.git
cd paperagent

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black paperagent/
flake8 paperagent/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent workflows
- [arXiv API](https://arxiv.org/help/api) - Open access to research papers
- [Scholarly](https://github.com/scholarly-python-package/scholarly) - Google Scholar API
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Streamlit](https://streamlit.io/) - Rapid UI development

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/bingdongni/paperagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bingdongni/paperagent/discussions)
- **Email**: 2905153124@qq.com

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/paperagent&type=Date)](https://star-history.com/#yourusername/paperagent&Date)

---

## 📜 Citation

If you use PaperAgent in your research, please cite:

```bibtex
@software{paperagent2024,
  title = {PaperAgent: Academic Multi-Agent Collaboration Framework},
  author = {bingdongni},
  year = {2026},
  url = {https://github.com/bingdongni/paperagent}
}
```

---

<div align="center">

**Built with ❤️ for the research community**

[⬆ Back to Top](#paperagent-academic-multi-agent-collaboration-framework)

</div>
