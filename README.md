📄 Document Intelligence System

A lightweight end-to-end Document Understanding Pipeline that processes business documents and produces structured insights.

This project extracts text from PDF/TXT files, cleans it, performs Named Entity Recognition (NER), runs document classification, generates summaries, and outputs everything as a structured JSON insights file.

✔ HR
✔ Legal
✔ Finance
✔ Medical
✔ Customer Support
✔ Unknown / Generic

🚀 Features
1. PDF/Text Extraction

Uses pdfplumber for PDF parsing

Graceful fallback for text files

Works with scanned PDFs when OCR is added

2. Text Cleaning & Normalization

Whitespace normalization

Header/footer cleanup

Sentence-safe preprocessing

3. Named Entity Recognition (NER)

Uses spaCy to extract:

ORG

PERSON

DATE

MONEY

GPE

and many others

4. Document Classification

Two modes:

Rule-based classifier (fast, no training needed)

Optional upgrade: sklearn / transformer-based classifier

5. Summarization

Two options:

Lightweight extractive summarizer (no heavy models)

Optional: Transformers-based abstractive summarization

6. Insights Builder

Generates structured JSON:

Entities and counts

Document category

Summary

Top keywords

File name and metadata

📦 Deliverables Included
doc_intel.py — standalone script

sample_docs/ — 5 labeled documents: HR, Legal, Finance, Medical, Support

sample_output.json — example output

🧠 Architecture (Pipeline)
PDF/TXT

   ↓
   
Extraction (pdfplumber)

   ↓
   
Cleaning & Normalization

   ↓
   
NER (spaCy)

   ↓
   
Document Classification (Rule-based / ML)

   ↓
   
Summarization (Extractive or LLM)

   ↓
   
Insights JSON


▶️ Quickstart

pip install -r requirements.txt
python doc_intel.py --input_dir sample_docs --output output.json

