"""
doc_intel.py
Document Intelligence pipeline with model caching, optional parallelism, safer PDF handling.

Usage:
    python doc_intel.py --input_dir sample_docs --output sample_output.json --workers 4
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Utilities ----
def safe_read_text(path: Path) -> str:
    """Try to read text files robustly. For PDFs, return empty and let pdf extractor handle them."""
    if path.suffix.lower() in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return path.read_text(encoding="latin-1", errors="ignore")
    return ""

# ---- Text extraction ----
def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber or fallback to PyPDF2. If both fail, raise."""
    if path.suffix.lower() in [".txt", ".md"]:
        return safe_read_text(path)

    try:
        import pdfplumber
        text_pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text_pages.append(page.extract_text() or "")
        return "\n\n".join(text_pages).strip()
    except Exception as e1:
        # fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n\n".join(pages).strip()
        except Exception as e2:
            # Give a clear error rather than returning binary
            print(f"[error] PDF extraction failed for {path.name}: pdfplumber err: {e1}; PyPDF2 err: {e2}", file=sys.stderr)
            return ""

# ---- Cleaning ----
def clean_text(text: str, preserve_paragraphs: bool = True) -> str:
    """Basic cleaning. If preserve_paragraphs True, keep double-newlines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_paragraphs:
        # collapse many blank lines to exactly one blank line, but keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # collapse repeated spaces but keep newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # normalize stray multiple newlines/space combos to single newline (but keep paragraphs)
        text = re.sub(r' *\n +', '\n', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---- NER (spaCy) ----
class NerRunner:
    def __init__(self, model_name: str = "en_core_web_sm", disable=[]):
        self.model_name = model_name
        self.nlp = None
        self.disable = disable

    def load(self):
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load(self.model_name, disable=self.disable)
            except Exception as e:
                print(f"[warn] spaCy model load failed ({self.model_name}): {e}. NER disabled.", file=sys.stderr)
                self.nlp = None

    def run(self, text: str) -> List[Tuple[str, str]]:
        self.load()
        if not self.nlp or not text:
            return []
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

# ---- Classification ----
def classify_text(text: str, classifier=None, keywords: Optional[Dict[str, List[str]]] = None) -> str:
    if classifier is not None:
        try:
            return classifier.predict([text])[0]
        except Exception as e:
            print(f"[warn] classifier predict failed: {e}", file=sys.stderr)

    if keywords is None:
        keywords = {
            "HR": ["employee", "hiring", "onboarding", "salary", "benefit", "leave"],
            "Legal": ["agreement", "party", "warranty", "liability", "clause", "hereby"],
            "Finance": ["invoice", "balance", "revenue", "expense", "profit", "transaction"],
            "Medical": ["patient", "diagnosis", "treatment", "prescription", "symptom"],
            "Support": ["ticket", "customer", "issue", "support", "resolution", "sla"]
        }
    tl = re.findall(r"\w+", text.lower())
    freq = Counter(tl)
    scores = {}
    for k, kws in keywords.items():
        scores[k] = sum(freq.get(w.lower(), 0) for w in kws)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Unknown"
    return best

# ---- Summarization (HuggingFace pipeline cached) ----
class Summarizer:
    def __init__(self, model_name: Optional[str] = None, tokenizer_chunk: int = 1000):
        """
        model_name: hf model for pipeline (if None -> use naive extractive fallback)
        tokenizer_chunk: approximate characters per chunk for model summarizer
        """
        self.model_name = model_name
        self.pipeline = None
        self.chunk_size = tokenizer_chunk

    def load(self):
        if self.pipeline is None and self.model_name:
            try:
                from transformers import pipeline
                self.pipeline = pipeline("summarization", model=self.model_name, truncation=True)
            except Exception as e:
                print(f"[warn] Summarizer pipeline load failed ({self.model_name}): {e}. Using fallback.", file=sys.stderr)
                self.pipeline = None

    def summarize(self, text: str, max_length: int = 150) -> str:
        if not text:
            return ""
        self.load()
        if self.pipeline:
            # naive chunking by characters so we don't exceed model limits
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            parts = []
            for c in chunks:
                try:
                    out = self.pipeline(c, max_length=max_length, min_length=30, do_sample=False)
                    parts.append(out[0].get('summary_text', '').strip())
                except Exception as e:
                    print(f"[warn] summarizer failed on a chunk: {e}", file=sys.stderr)
            return " ".join([p for p in parts if p])
        # fallback extractive: pick top 3 sentences by frequency score
        sents = re.split(r'(?<=[.!?])\s+', text)
        if not sents:
            return ""
        words = re.findall(r'\w+', text.lower())
        freq = Counter(words)
        def score(sent):
            tokens = re.findall(r'\w+', sent.lower())
            if not tokens:
                return 0.0
            return sum(freq.get(t,0) for t in tokens) / (len(tokens) + 1)
        scored = sorted(sents, key=score, reverse=True)
        return " ".join(scored[:3])

# ---- Insights ----
def extract_insights(ent_list: List[Tuple[str, str]], summary: str, text: str) -> Dict[str, Any]:
    c = Counter([label for _, label in ent_list])
    top_terms = Counter(re.findall(r'\w+', text.lower())).most_common(20)
    return {
        "entity_counts": dict(c),
        "top_terms": top_terms,
        "summary": summary
    }

# ---- File processing ----
def process_file(path: Path, ner_runner: NerRunner, summarizer: Summarizer, classifier=None, preserve_paragraphs=True) -> Dict[str, Any]:
    raw = ""
    if path.suffix.lower() in (".txt", ".md"):
        raw = safe_read_text(path)
    else:
        raw = extract_text_from_pdf(path)
    cleaned = clean_text(raw, preserve_paragraphs=preserve_paragraphs)
    ner = ner_runner.run(cleaned) if ner_runner else []
    category = classify_text(cleaned, classifier)
    summary = summarizer.summarize(cleaned) if summarizer else ""
    insights = extract_insights(ner, summary, cleaned)
    return {
        "file": str(path),
        "category": category,
        "entities": ner,
        "summary": summary,
        "insights": insights
    }

# ---- CLI / main ----
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="sample_output.json")
    parser.add_argument("--ner_model", default="en_core_web_sm")
    parser.add_argument("--summ_model", default=None, help="HF summarization model (optional). If omitted an extractive fallback is used.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel threads. Use 1 for sequential.")
    parser.add_argument("--preserve_paragraphs", action="store_true")
    parser.add_argument("--file_types", default=".pdf,.txt,.md", help="Comma-separated list of file extensions to process")
    args = parser.parse_args(argv)

    in_path = Path(args.input_dir)
    if not in_path.exists() or not in_path.is_dir():
        print(f"[error] input_dir {in_path} not found or not a directory", file=sys.stderr)
        return

    allowed = {ext.strip().lower() if ext.strip().startswith('.') else '.' + ext.strip().lower()
               for ext in args.file_types.split(',')}
    files = sorted([p for p in in_path.iterdir() if p.suffix.lower() in allowed])

    ner_runner = NerRunner(model_name=args.ner_model)
    summarizer = Summarizer(model_name=args.summ_model)

    results = []
    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, p, ner_runner, summarizer, None, args.preserve_paragraphs): p for p in files}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results.append(fut.result())
                    print(f"[ok] processed {p.name}")
                except Exception as e:
                    print(f"[error] failed processing {p.name}: {e}", file=sys.stderr)
    else:
        for p in files:
            print(f"Processing {p.name} ...")
            try:
                res = process_file(p, ner_runner, summarizer, None, args.preserve_paragraphs)
                results.append(res)
            except Exception as e:
                print(f"[error] failed processing {p.name}: {e}", file=sys.stderr)

    # Write JSON (ensure ASCII safe)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} records to {args.output}")

if __name__ == "__main__":
    main()
