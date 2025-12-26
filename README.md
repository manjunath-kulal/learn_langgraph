# 🤖 Lang-Recruit

AI-powered recruitment agent built with **LangGraph** and **LangChain** that autonomously screens candidates, grades resumes, and makes hiring decisions with human-in-the-loop capabilities.

## Features

- Semantic resume search with FAISS vector database
- AI-powered candidate scoring and ranking
- Human-in-the-loop review for borderline candidates
- Dynamic routing (PASS/REVIEW/REJECT)
- Conversational interface for natural language queries

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirments.txt

# Set up environment
echo "GEMINI_API_KEY=your_key_here" > .env

# Ingest resumes
python ingest.py
```

### Usage

```bash
python main.py
```

Example:
```
Recruiter: Find me a senior Python developer with ML experience

 **Top Candidate Found!**
 **Name:** John Doe
 **Match Score:** 95/100
```

## Project Structure

```
├── main.py            # Main application & LangGraph workflow
├── graph_nodes.py     # Node implementations (analyze, retrieve, grade)
├── ingest.py          # Resume ingestion pipeline
├── resumes/           # PDF resume storage
└── resume_index/      # FAISS vector database
```

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS

## Routing Logic

| Decision | Score | Action |
|----------|-------|--------|
| DIRECT_PASS | 90+ | Schedule Interview |
| HUMAN_NEEDED | 70-89 | Manual Review |
| REJECT | <70 | End |

---

Built with LangGraph, LangChain, and Google Gemini
