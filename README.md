

<h1 align="center">📄 Document Intelligence System</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/spaCy-NLP-green" />
  <img src="https://img.shields.io/github/stars/ravikiranc713-dotcom/Document-IntelligenceSystem" />
  <img src="https://img.shields.io/github/issues/ravikiranc713-dotcom/Document-IntelligenceSystem" />
  <img src="https://img.shields.io/github/last-commit/ravikiranc713-dotcom/Document-IntelligenceSystem" />
</p>
<p align="center">A lightweight end-to-end Document Understanding Pipeline.</p>


---

# 📄 Document Intelligence System

A lightweight, production-ready Python pipeline for **document ingestion, text extraction, cleaning, NER, classification, summarization, and insight generation**.
Supports **PDF, TXT, MD**, with optional **parallel processing** and **model caching** for performance.

---

## 🚀 Features

### ✔️ Robust Document Extraction

* Extracts text from PDF using **pdfplumber → PyPDF2 (fallback)**
* Gracefully handles failures and corrupt PDFs
* Supports `.txt` and `.md` directly

### ✔️ Clean, Well-Structured Text Processing

* Paragraph-preserving cleaning
* Whitespace normalization
* Optional fully-flattened cleaning mode

### ✔️ NLP Intelligence

* **Named Entity Recognition (NER)** via spaCy
* **Keyword-based text classification** (HR, Legal, Finance, Medical, Support)
* **Summarization** using:

  * HuggingFace `pipeline` (if model provided)
  * OR fallback naive extractive summarizer

### ✔️ Insights Module

* Entity distribution
* Top term frequencies
* Per-document structured metadata

### ✔️ Fast Parallel Processing

* ThreadPoolExecutor with **safe model preloading**
* Configurable worker count (`--workers N`)

### ✔️ Clean Logging & Error Handling

* Uses Python `logging` for professional output
* Skips unreadable / empty docs gracefully

---

## 📦 Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Recommended optional packages

```bash
pip install pdfplumber PyPDF2 spacy transformers
python -m spacy download en_core_web_sm
```

---

## 🧠 Usage

### Basic usage

```bash
python doc_intel.py --input_dir sample_docs --output results.json
```

### With summarization model + 4 worker threads

```bash
python doc_intel.py \
  --input_dir documents \
  --output output.json \
  --summ_model facebook/bart-large-cnn \
  --workers 4
```

### Preserving paragraph structure (recommended for contracts/emails)

```bash
python doc_intel.py --input_dir docs --preserve_paragraphs
```

### Supported file types

* `.pdf`
* `.txt`
* `.md`

Control via:

```bash
--file_types ".pdf,.txt"
```

---

## 📁 Output Format (JSON)

Each document produces structured metadata:

```json
{
  "file": "sample.pdf",
  "category": "Legal",
  "entities": [
    ["OpenAI", "ORG"],
    ["2024", "DATE"]
  ],
  "summary": "Short summary text...",
  "insights": {
    "entity_counts": {
      "ORG": 2,
      "DATE": 1
    },
    "top_terms": [["contract", 12], ["party", 9]],
    "summary": "Short summary..."
  }
}
```

---

## ⚙️ Command-Line Arguments

| Argument                | Description                       | Default              |
| ----------------------- | --------------------------------- | -------------------- |
| `--input_dir`           | Directory containing documents    | **Required**         |
| `--output`              | Output JSON file                  | `sample_output.json` |
| `--ner_model`           | spaCy model name                  | `en_core_web_sm`     |
| `--summ_model`          | HuggingFace model name (optional) | None                 |
| `--workers`             | Number of parallel threads        | 1                    |
| `--preserve_paragraphs` | Keep paragraph structure          | Off                  |
| `--file_types`          | Comma-separated allowed types     | `.pdf,.txt,.md`      |

---

## 🧩 Architecture

```
┌──────────────┐
│   Ingestion   │  ← PDFs, TXT, MD
└──────┬───────┘
       ↓
┌──────────────┐
│ Text Extract  │  (pdfplumber → PyPDF2 fallback)
└──────┬───────┘
       ↓
┌──────────────┐
│   Cleaning    │  (whitespace, paragraphs)
└──────┬───────┘
       ↓
┌──────────────┐
│ NLP Pipeline  │─ NER (spaCy)  
│               │─ Classifier  
│               │─ Summarizer (HF / fallback)
└──────┬───────┘
       ↓
┌──────────────┐
│   Insights    │  (top terms, entity counts)
└──────┬───────┘
       ↓
┌──────────────┐
│   JSON Out    │
└──────────────┘
```

---



