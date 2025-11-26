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

Note: spaCy models are not included in the requirements.txt because they are packaged separately. Make sure to download the English NER model before running the script.

Uses spaCy to extract:

ORG

PERSON

DATE

MONEY

GPE

and many others

📦 spaCy Model Requirement

This project uses spaCy for Named Entity Recognition (NER).
spaCy requires a separate language model, which is not included by default.

After installing dependencies, install the English model:

python -m spacy download en_core_web_sm

What this model does in this project?

The en_core_web_sm model enables:

Entity extraction such as:

ORGANIZATION (ORG)

PERSON

MONEY

DATE

GPE (locations)

etc.

It allows the Document Intelligence System to detect key mentions like:

Employee names

Company names

Legal terms

Financial values

Medical terms

Customer names

Locations and dates

Example of entities extracted:

[("Acme Corp", "ORG"), ("₹50,000", "MONEY"), ("2023", "DATE")]


If the model is not installed, the program will continue running but will log a warning and return empty NER results.

## Named Entity Recognition Example

Input:
"Acme Corp agreed to pay ₹50,000 by 15 March 2023."

Entities:
- ORG: Acme Corp  
- MONEY: ₹50,000  
- DATE: 15 March 2023  


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

