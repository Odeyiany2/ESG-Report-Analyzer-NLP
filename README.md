# 🌿 ESG Report Analyzer

An AI-powered tool that analyzes corporate sustainability reports against global ESG standards — detecting compliance gaps, vague disclosures, and missing sections through a conversational interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-4B8BBE?style=flat)

---

## Overview

Companies produce ESG reports that run into hundreds of pages. Auditing them manually against frameworks like GRI, SASB, and IFRS S1/S2 is time-consuming, inconsistent, and prone to oversight.

This tool lets a company upload its sustainability report and instantly receive a structured analysis that:

- Classifies content into **Environmental**, **Social**, and **Governance** categories
- Identifies **vague or unsubstantiated disclosures** that lack measurable data
- Flags **compliance gaps and missing sections** relative to GRI, SASB, and IFRS standards
- Provides **confidence scores** and **key token explanations** for each classification
- Allows **conversational follow-up** — ask questions about your own report in plain language

---

## Architecture

```
ESG-Report-Analyzer/
└── ESG-Report-Analyzer-NLP/
    ├── src/
    │   ├── agentic_ai/         # LangGraph orchestration pipeline
    │   │   └── orchestrator.py
    │   ├── backend/            # FastAPI application
    │   │   └── fastapi_logic.py
    │   ├── frontend/           # Streamlit UI
    │   │   └── app.py
    │   ├── nlp/                # Core NLP pipeline
    │   │   ├── doc_handler.py  # Document ingestion (PDF, DOCX, TXT)
    │   │   ├── embeddings.py   # Embedding + Chroma vector store
    │   │   ├── retriever.py    # RAG retrieval, FinBERT classification, LLM analysis
    │   │   └── prompt_esg.yml  # Modular prompt templates
    │   ├── models/             # ML model assets
    │   └── utils/              # Logging and shared utilities
    ├── data/
    │   └── standards/          # GRI, SASB, IFRS reference documents (not included)
    ├── vectorstores/           # Chroma vector store persistence (auto-generated)
    ├── tests/
    ├── notebook/
    ├── .env
    └── requirements.txt
```

### Pipeline Flow

```
User uploads ESG report
        │
        ▼
 Document Ingestion          ← doc_handler.py (PDF / DOCX / TXT)
        │
        ▼
 Embedding + Indexing        ← embeddings.py (e5-base-v2 + Chroma)
        │
        ▼
 ┌─────────────────────────────────────────┐
 │         LangGraph Agent Pipeline        │
 │                                         │
 │  [Retrieve] → [Classify] → [Summarize]  │
 │                                         │
 │  Retrieve:  similarity search against   │
 │             standards + report stores   │
 │                                         │
 │  Classify:  FinBERT-ESG (E / S / G /   │
 │             None) + attention-based     │
 │             explainability              │
 │                                         │
 │  Summarize: prompt assembly → LLM →    │
 │             structured analysis output  │
 └─────────────────────────────────────────┘
        │
        ▼
 Structured Analysis + Follow-up Chat     ← Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document ingestion | LangChain document loaders (PyPDF, Docx2txt, TextLoader) |
| Embeddings | `intfloat/e5-base-v2` via HuggingFace |
| Vector store | ChromaDB (persistent, session-scoped) |
| ESG classification | `yiyanghkust/finbert-esg` (4-class: E / S / G / None) |
| Explainability | Attention weight extraction (last transformer layer) |
| Agentic pipeline | LangGraph (StateGraph with MemorySaver checkpointing) |
| LLM | GPT via HuggingFace Inference Router |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Prompt management | YAML-based modular prompt templates |

---

## Features

- **Dual vector store retrieval** — standards and uploaded reports are indexed separately, then retrieved together to ground the analysis in both the company's disclosures and the relevant framework requirements
- **FinBERT-ESG classification with explainability** — each retrieved section is classified and the top 5 attention tokens driving the classification are surfaced
- **Agentic workflow** — LangGraph manages state across retrieve → classify → summarize nodes with conversation memory per session
- **Session-scoped analysis** — each upload gets a unique session ID; multiple users can run concurrent analyses without interference
- **Conversational follow-up** — after the initial structured analysis, users can ask plain-language questions about their report
- **Downloadable report** — the full analysis exports as a `.md` file

---

## Setup

### Prerequisites

- Python 3.10+
- A HuggingFace API key (for the LLM via the HuggingFace Inference Router)
- GRI, SASB, and/or IFRS standard documents in PDF/DOCX/TXT format

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ESG-Report-Analyzer.git
cd ESG-Report-Analyzer/ESG-Report-Analyzer-NLP
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
STANDARD_DOCS_DIR=data/standards
STANDARDS_VECTOR_PATH=vectorstores/standards
REPORTS_VECTOR_PATH=vectorstores/uploaded_reports
PROMPTS_FILE_PATH=src/nlp/prompt_esg.yml
```

### 5. Add ESG standard documents

Place your GRI, SASB, and/or IFRS reference documents (PDF, DOCX, or TXT) into `data/standards/`. You can organise them in subfolders by framework:

```
data/standards/
├── GRI/
├── SASB/
└── IFRS/
```

> The standards vector store is built automatically on first run and cached for subsequent sessions.

### 6. Run the FastAPI backend

```bash
uvicorn src.backend.fastapi_logic:app --reload --port 8000
```

### 7. Run the Streamlit frontend

Open a second terminal (with the virtual environment activated):

```bash
streamlit run src/frontend/app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Open the app in your browser
2. Upload a sustainability report (PDF, DOCX, or TXT) using the left panel
3. Click **Run ESG Analysis**
4. Review the structured analysis — coverage summary, percept table, compliance assessment, and identified gaps
5. Ask follow-up questions using the chat input
6. Download the full report as a `.md` file from the left panel

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms API is running and standards documents are loaded |
| `POST` | `/upload` | Upload ESG report files; returns a `session_id` |
| `POST` | `/query` | Submit a query against an uploaded report |
| `DELETE` | `/session/{session_id}` | Clear session data from memory |

---

## Limitations & Known Gaps

- **FinBERT token limit** — FinBERT processes up to 512 tokens per chunk. Chunks exceeding this are safely truncated before classification, which may reduce classification accuracy on very long paragraphs.
- **In-memory session store** — uploaded documents are held in server memory and reset on restart. Not suitable for multi-user production deployment without a persistent store (e.g. Redis).
- **Standard document quality** — analysis quality depends on the completeness of the GRI/SASB/IFRS reference documents you provide. More complete standard documents produce more precise gap detection.
- **LLM dependency** — the summarization step requires a live HuggingFace Inference Router connection. Offline use is not currently supported.

---

## Roadmap

- [ ] ML-based ESG score prediction layer
- [ ] Support for multi-report comparison (year-on-year analysis)
- [ ] Expanded framework coverage (TCFD, CDP, ESRS)
- [ ] Production-ready deployment with Redis session store
- [ ] Automated test suite

---

## Disclaimer

This tool is intended to assist ESG analysis and should not be used as a substitute for professional audit or compliance review. Classifications and gap assessments are AI-generated and should be verified by a qualified practitioner.

---

## Author

Built by Miriam — combining accounting, NLP, and agentic AI to make ESG reporting more transparent and auditable.

Connect on [LinkedIn]([https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/miriam-itopa-odeyiany-919787245)) · 
