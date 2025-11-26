
"""
doc_intel.py
Standalone pipeline for Document Intelligence System.

Usage:
    python doc_intel.py --input_dir sample_docs --output sample_output.json
"""
import argparse, os, json, re
from pathlib import Path
from collections import Counter

# Optional imports (import within functions for graceful failure)
def extract_text_from_pdf(path):
    """Placeholder: try to extract text from PDF, fallback to reading .txt"""
    p = Path(path)
    if p.suffix.lower() in ['.txt']:
        return p.read_text(encoding='utf-8', errors='ignore')
    try:
        import pdfplumber
        text = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        print(f"[warn] pdfplumber not available or failed for {p.name}: {e}")
        return p.read_text(encoding='utf-8', errors='ignore')

def clean_text(text):
    # basic cleaning
    text = text.replace("\r\n", "\n")
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_ner(text, model_name="en_core_web_sm"):
    """Run spaCy NER (returns list of (ent_text, ent_label))"""
    try:
        import spacy
        nlp = spacy.load(model_name)
    except Exception as e:
        print(f"[warn] spaCy model load failed: {e}. Returning empty NER.")
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def classify_text(text, classifier=None):
    """Simple rule-based classifier fallback.
    classifier param can be a sklearn model with predict method.
    """
    if classifier:
        try:
            return classifier.predict([text])[0]
        except Exception as e:
            print("[warn] classifier predict failed:", e)
    # simple keyword based classification (HR / Legal / Finance / Medical / Support)
    keywords = {
        "HR": ["employee", "hiring", "onboarding", "salary", "benefit", "leave"],
        "Legal": ["agreement", "party", "warranty", "liability", "clause", "hereby"],
        "Finance": ["invoice", "balance", "revenue", "expense", "profit", "transaction"],
        "Medical": ["patient", "diagnosis", "treatment", "prescription", "symptom"],
        "Support": ["ticket", "customer", "issue", "support", "resolution", "sla"]
    }
    scores = {k:0 for k in keywords}
    tl = text.lower()
    for k, kws in keywords.items():
        for kw in kws:
            scores[k] += tl.count(kw)
    # return best or Unknown
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Unknown"
    return best

def summarize_text(text, max_length=150):
    """Use HuggingFace transformers summarization pipeline if available, otherwise naive summarization."""
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        parts = []
        for c in chunks:
            out = summarizer(c, max_length=max_length, min_length=30, do_sample=False)
            parts.append(out[0]['summary_text'])
        return " ".join(parts)
    except Exception as e:
        # naive extractive summary: top sentences by word overlap
        sents = re.split(r'(?<=[.!?]) +', text)
        if not sents:
            return ""
        # simple scoring by word frequency
        words = re.findall(r'\w+', text.lower())
        freq = Counter(words)
        def score(sent):
            return sum(freq.get(w.lower(),0) for w in re.findall(r'\w+', sent)) / (len(sent.split())+1)
        scored = sorted(sents, key=score, reverse=True)
        return " ".join(scored[:3])

def extract_insights(ent_list, summary, text):
    insights = {}
    # entity counts
    c = Counter([label for _, label in ent_list])
    insights['entity_counts'] = dict(c)
    insights['top_terms'] = Counter(re.findall(r'\w+', text.lower())).most_common(20)
    insights['summary'] = summary
    return insights

def process_file(path, classifier=None):
    text = extract_text_from_pdf(path)
    cleaned = clean_text(text)
    ner = run_ner(cleaned)
    category = classify_text(cleaned, classifier)
    summary = summarize_text(cleaned)
    insights = extract_insights(ner, summary, cleaned)
    return {
        "file": str(path),
        "category": category,
        "entities": ner,
        "summary": summary,
        "insights": insights
    }

def main(args):
    input_dir = Path(args.input_dir)
    out = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() not in ['.pdf', '.txt', '.md']:
            continue
        print("Processing", p.name)
        out.append(process_file(p))
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote output to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="sample_output.json")
    args = parser.parse_args()
    main(args)
