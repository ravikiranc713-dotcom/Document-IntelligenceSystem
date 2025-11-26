

---

# 📄 Document Intelligence System

A lightweight end-to-end **Document Understanding Pipeline** that processes business documents and produces structured insights.

This project extracts text from PDF/TXT files, cleans it, performs **Named Entity Recognition (NER)**, runs **document classification**, generates **summaries**, and outputs everything as a structured **JSON insights file**.

Supported document types:

✔ HR
✔ Legal
✔ Finance
✔ Medical
✔ Customer Support
✔ Unknown / Generic

---

# 🚀 Features

## **1. PDF/Text Extraction**

* Uses `pdfplumber` for PDF parsing
* Graceful fallback for `.txt` files
* Works with scanned PDFs when OCR is added later

---

## **2. Text Cleaning & Normalization**

* Whitespace normalization
* Header/footer cleanup
* Sentence-safe preprocessing

---

## **3. Named Entity Recognition (NER)**

> **Note:** spaCy models are not included in `requirements.txt` because they are packaged separately.
> You must download the English NER model before running the pipeline.

This project uses **spaCy** to extract entities such as:

* ORG (Organizations)
* PERSON
* DATE
* MONEY
* GPE (Locations)
* Many more

### 📦 **spaCy Model Requirement**

After installing dependencies, install the English model:

```bash
python -m spacy download en_core_web_sm
```

### **What this model does in this project?**

The `en_core_web_sm` model enables extraction of semantic information like:

* Employee names
* Company names
* Legal terms
* Financial values
* Medical terminology
* Customer names
* Locations and dates

### **Example**

**Input:**

```
"Acme Corp agreed to pay ₹50,000 by 15 March 2023."
```

**Entities Extracted:**

* **ORG:** Acme Corp
* **MONEY:** ₹50,000
* **DATE:** 15 March 2023

If the model is *not* installed, the program still runs but logs a warning and returns empty NER results.

---

## **4. Document Classification**

Two modes:

* ✔ Rule-based classifier (fast, no training needed)
* ✔ Optional ML classifier (sklearn or transformers)

---

## **5. Summarization**

Two options:

* ✔ Lightweight extractive summarizer (default; no heavy models)
* ✔ Optional abstractive summarization (transformers-based)

---

## **6. Insights Builder**

Generates structured JSON output including:

* Extracted entities
* Entity counts
* Document category
* Summary
* Top keywords
* Metadata

---

# 📦 Deliverables Included

* **`doc_intel.py`** — standalone pipeline script
* **`sample_docs/`** — 5 labeled business documents (HR, Legal, Finance, Medical, Support)
* **`sample_output.json`** — example output JSON
* *(Optional)* `notebook/Document_Intelligence.ipynb` — interactive walkthrough
* *(Optional)* `WHITEBOARD.md` — system design + interview notes

---

# 🧠 Architecture (Pipeline)

```
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
```

---

# ▶️ Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Run the pipeline:

```bash
python doc_intel.py --input_dir sample_docs --output output.json
```

---

