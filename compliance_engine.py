"""
ONGC Compliance Matrix Evaluation Engine
=========================================
Evaluates vendor documents against every clause in the compliance matrix.

Architecture
------------
  compliance_matrix.xlsx  (read directly — no OCR)
      ↓
  Per-clause requirement string
      ↓
  Vendor-filtered ChromaDB query  →  top 15 chunks (only that vendor's docs)
      ↓
  BM25 keyword search over vendor chunks
      ↓
  Hybrid-ranked evidence pool
      ↓
  Qwen2.5 via Ollama  →  JSON verdict {status, explanation, pages, sources}
      ↓
  compliance_results/{vendor}_results.json  (raw, used by report generator)

Usage
-----
    # Evaluate ALL vendors:
    python compliance_engine.py

    # Evaluate a single vendor:
    python compliance_engine.py --vendor Vendor_A

    # Dry-run (retrieve only, skip LLM):
    python compliance_engine.py --dry-run

Place this file at:
    ONGC_RAG/compliance_engine.py

Prerequisites
-------------
    pip install pandas openpyxl rank_bm25 chromadb sentence-transformers requests
    ollama pull qwen2.5:7b   (or 3b for faster inference)
    ollama serve
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    sys.exit("Run: pip install pandas openpyxl")

try:
    import requests
except ImportError:
    sys.exit("Run: pip install requests")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("Run: pip install rank_bm25")

# =====================================
# PATHS  (all relative to ONGC_RAG/)
# =====================================
MATRIX_FILE       = Path("compliance_matrix") / "compliance_matrix.xlsx"
VECTOR_STORE_DIR  = Path("vector_store")
CHUNKS_FILE       = Path("chunks") / "chunks.json"
RESULTS_DIR       = Path("compliance_results")

# =====================================
# CONFIG
# =====================================
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "qwen2.5:7b"          # change to :3b for speed / less RAM

EMBED_MODEL       = "all-MiniLM-L6-v2"
VECTOR_TOP_K      = 15                    # vector candidates per clause
BM25_TOP_K        = 10                    # BM25 candidates per clause
FINAL_TOP_K       = 15                    # after hybrid merge, pass this many to LLM

COLLECTION_TEXT   = "tender_text"
COLLECTION_TABLE  = "tender_tables"

# Compliance verdict labels
STATUS_COMPLIANT     = "COMPLIANT"
STATUS_NON_COMPLIANT = "NON_COMPLIANT"
STATUS_NOT_FOUND     = "NOT_FOUND"

# Expected compliance matrix column names (case-insensitive matching applied)
COL_CLAUSE      = "clause no"
COL_REQUIREMENT = "requirement"
COL_REMARKS     = "remarks"


# =====================================
# SYSTEM PROMPT FOR EVALUATION
# =====================================
EVAL_SYSTEM_PROMPT = """You are a strict technical compliance auditor for oil & gas equipment tenders.

Your job: evaluate whether vendor-supplied evidence satisfies a specific requirement.

RULES (non-negotiable):
1. Base verdict ONLY on the evidence chunks provided. Never invent facts.
2. If any evidence chunk explicitly satisfies the requirement → COMPLIANT.
3. If any evidence chunk explicitly contradicts the requirement → NON_COMPLIANT.
4. If no chunk mentions the requirement topic at all → NOT_FOUND.
5. For numerical specs (voltage, power, rpm, etc.) — compare exact values.
   A value ±5% of the required value is still COMPLIANT unless otherwise stated.
6. Do NOT infer compliance from vague or unrelated text.
7. Always list the exact page numbers and document names where you found evidence.

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{
  "status": "COMPLIANT" | "NON_COMPLIANT" | "NOT_FOUND",
  "explanation": "One or two sentences explaining the verdict.",
  "evidence_pages": [list of integer page numbers],
  "evidence_sources": [list of source document filenames],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "ocr_warning": true | false
}"""


# =====================================
# OLLAMA CLIENT
# =====================================
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def evaluate(self, requirement: str, evidence_block: str) -> dict:
        """
        Send requirement + evidence to Qwen and parse the JSON verdict.
        Returns a dict with status, explanation, pages, sources.
        Falls back to NOT_FOUND on any parsing error.
        """
        user_msg = (
            f"REQUIREMENT:\n{requirement}\n\n"
            f"EVIDENCE FROM VENDOR DOCUMENTS:\n{evidence_block}\n\n"
            f"Evaluate compliance. Return only the JSON object."
        )

        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "stream":  False,
            "options": {
                "temperature": 0.05,   # near-zero for deterministic verdicts
                "top_p":       0.9,
                "num_ctx":     8192,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw_text = resp.json()["message"]["content"].strip()

            # Strip markdown code fences if present
            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

            return json.loads(raw_text)

        except json.JSONDecodeError:
            # LLM returned non-JSON — extract partial info with regex
            return self._parse_fallback(raw_text if 'raw_text' in dir() else "")
        except Exception as e:
            return {
                "status":           STATUS_NOT_FOUND,
                "explanation":      f"LLM call failed: {str(e)}",
                "evidence_pages":   [],
                "evidence_sources": [],
                "confidence":       "LOW",
                "ocr_warning":      False,
            }

    def _parse_fallback(self, text: str) -> dict:
        """Emergency parser when LLM returns partial JSON or prose."""
        status = STATUS_NOT_FOUND
        if re.search(r"\bCOMPLIANT\b", text, re.IGNORECASE) and \
           not re.search(r"\bNON.COMPLIANT\b", text, re.IGNORECASE):
            status = STATUS_COMPLIANT
        elif re.search(r"\bNON.COMPLIANT\b", text, re.IGNORECASE):
            status = STATUS_NON_COMPLIANT

        pages = [int(p) for p in re.findall(r"\bpage\s+(\d+)\b", text, re.IGNORECASE)]

        return {
            "status":           status,
            "explanation":      text[:300] if text else "Could not parse LLM response.",
            "evidence_pages":   pages,
            "evidence_sources": [],
            "confidence":       "LOW",
            "ocr_warning":      False,
        }


# =====================================
# MATRIX LOADER
# =====================================
def load_compliance_matrix(path: Path) -> list[dict]:
    """
    Read compliance_matrix.xlsx.
    Handles column name variations (case-insensitive, strip spaces).
    Returns list of {clause_no, requirement, remarks}.
    """
    if not path.exists():
        sys.exit(f"ERROR: Compliance matrix not found at {path}\n"
                 f"Expected: {path.resolve()}")

    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Flexible column matching
    def find_col(keyword: str) -> Optional[str]:
        for c in df.columns:
            if keyword in c:
                return c
        return None

    clause_col  = find_col("clause")
    req_col     = find_col("requirement") or find_col("description") or find_col("spec")
    remarks_col = find_col("remark")

    if not clause_col or not req_col:
        sys.exit(
            f"ERROR: compliance_matrix.xlsx must have columns containing "
            f"'clause' and 'requirement'.\nFound columns: {list(df.columns)}"
        )

    rows = []
    for _, row in df.iterrows():
        clause  = str(row.get(clause_col, "")).strip()
        req     = str(row.get(req_col,    "")).strip()
        remarks = str(row.get(remarks_col, "")).strip() if remarks_col else ""

        # Skip empty or header-like rows
        if not clause or not req or clause.lower() in ("clause no", "nan", ""):
            continue
        if req.lower() in ("requirement", "nan", "description", ""):
            continue

        rows.append({
            "clause_no":   clause,
            "requirement": req,
            "remarks":     remarks if remarks != "nan" else "",
        })

    print(f"Loaded {len(rows)} requirements from compliance matrix.")
    return rows


# =====================================
# VENDOR-FILTERED RETRIEVER
# =====================================
class VendorRetriever:
    """
    Hybrid retriever (vector + BM25) restricted to ONE vendor's chunks.
    Prevents cross-vendor evidence contamination.
    """

    def __init__(self, vendor_name: str):
        self.vendor_name = vendor_name

        # ── Load embedding model ───────────────────────────────────────────────
        print(f"  Loading embedding model ({EMBED_MODEL})…")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        # ── Connect to ChromaDB ────────────────────────────────────────────────
        if not VECTOR_STORE_DIR.exists():
            sys.exit(f"ERROR: Vector store not found at {VECTOR_STORE_DIR.resolve()}\n"
                     f"Run run_pipeline.py first.")

        self.chroma = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.col_text  = self.chroma.get_collection(COLLECTION_TEXT)
            self.col_table = self.chroma.get_collection(COLLECTION_TABLE)
        except Exception as e:
            sys.exit(f"ERROR: ChromaDB collections not found. Run embed_store.py first.\n{e}")

        # ── Load BM25 corpus (vendor-filtered) ────────────────────────────────
        print(f"  Building BM25 index for vendor '{vendor_name}'…")
        self._vendor_chunks = self._load_vendor_chunks()
        self._bm25 = self._build_bm25()

        print(f"  Vendor '{vendor_name}': {len(self._vendor_chunks)} chunks indexed.")

    def _load_vendor_chunks(self) -> list[dict]:
        """Load only chunks belonging to this vendor from chunks.json."""
        if not CHUNKS_FILE.exists():
            sys.exit(f"ERROR: {CHUNKS_FILE} not found. Run chunking pipeline first.")

        with open(CHUNKS_FILE, encoding="utf-8") as f:
            all_chunks = json.load(f)

        # Filter by vendor — chunk must have vendor field set by run_pipeline
        vendor_chunks = [
            c for c in all_chunks
            if c.get("vendor", "").lower() == self.vendor_name.lower()
        ]

        if not vendor_chunks:
            print(f"  WARNING: No chunks found for vendor '{self.vendor_name}'.")
            print(f"  Available vendors: {list({c.get('vendor','') for c in all_chunks})}")

        return vendor_chunks

    def _build_bm25(self) -> Optional[BM25Okapi]:
        if not self._vendor_chunks:
            return None
        tokenized = [c["text"].lower().split() for c in self._vendor_chunks]
        return BM25Okapi(tokenized)

    def _vector_search(self, query: str, n: int) -> list[dict]:
        """Vector search restricted to this vendor via metadata filter."""
        q_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        vendor_filter = {"vendor": self.vendor_name}
        hits = []

        for col in (self.col_text, self.col_table):
            if col.count() == 0:
                continue
            try:
                res = col.query(
                    query_embeddings=[q_vec],
                    n_results=min(n, col.count()),
                    where=vendor_filter,
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(
                    res["documents"][0],
                    res["metadatas"][0],
                    res["distances"][0],
                ):
                    score = round(1 - dist, 4)
                    hits.append({
                        "text":            doc,
                        "score":           score,
                        "page":            meta.get("page", 0),
                        "section":         meta.get("section", ""),
                        "chunk_type":      meta.get("chunk_type", "text"),
                        "source_file":     meta.get("source_file", ""),
                        "vendor":          meta.get("vendor", self.vendor_name),
                        "tess_confidence": meta.get("tess_confidence", 0),
                        "easy_confidence": meta.get("easy_confidence", 0),
                    })
            except Exception:
                continue

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:n]

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        """BM25 keyword search over vendor chunks."""
        if not self._bm25 or not self._vendor_chunks:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n]

        hits = []
        for idx, score in ranked:
            if score <= 0:
                break
            c = self._vendor_chunks[idx]
            hits.append({
                "text":            c["text"],
                "score":           round(float(score) / 10.0, 4),  # normalise roughly
                "page":            c.get("page", 0),
                "section":         c.get("section", ""),
                "chunk_type":      c.get("chunk_type", "text"),
                "source_file":     c.get("source_file", ""),
                "vendor":          c.get("vendor", self.vendor_name),
                "tess_confidence": c.get("tess_confidence", 0),
                "easy_confidence": c.get("easy_confidence", 0),
            })
        return hits

    def search(self, query: str, n_results: int = FINAL_TOP_K) -> list[dict]:
        """
        Hybrid search: merge vector + BM25, deduplicate, return top n.
        All results are guaranteed to be from this vendor only.
        """
        vec_hits = self._vector_search(query, VECTOR_TOP_K)
        bm25_hits = self._bm25_search(query, BM25_TOP_K)

        # Merge by text identity
        seen_texts: set = set()
        merged: list[dict] = []

        for h in vec_hits + bm25_hits:
            key = h["text"][:120]   # first 120 chars as dedup key
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(h)

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:n_results]


# =====================================
# EVIDENCE FORMATTER
# =====================================
def format_evidence_block(chunks: list[dict]) -> str:
    """
    Build a numbered evidence block to pass to the LLM.
    Keeps only the most relevant text and metadata.
    """
    if not chunks:
        return "No relevant evidence found in vendor documents."

    lines = []
    for i, c in enumerate(chunks, 1):
        page    = c.get("page", "?")
        section = c.get("section", "General") or "General"
        source  = c.get("source_file", "unknown")
        ctype   = c.get("chunk_type", "text")
        tconf   = c.get("tess_confidence", 0)
        econf   = c.get("easy_confidence", 0)
        conf    = max(tconf, econf)
        score   = c.get("score", 0)
        text    = c.get("text", "").strip()

        conf_tag = f" ⚠LOW_OCR({conf:.0f}%)" if 0 < conf < 70 else ""
        lines.append(
            f"[{i}] Doc: {source} | Page {page} | {section} | "
            f"Type: {ctype} | Relevance: {score:.3f}{conf_tag}"
        )
        lines.append(text[:600])   # cap at 600 chars per chunk to fit context window
        lines.append("")

    return "\n".join(lines)


# =====================================
# COMPLIANCE EVALUATION ENGINE
# =====================================
class ComplianceEngine:
    def __init__(self, vendor_name: str, dry_run: bool = False):
        self.vendor_name = vendor_name
        self.dry_run     = dry_run

        print(f"\n{'='*60}")
        print(f"Initialising engine for vendor: {vendor_name}")
        print(f"{'='*60}")

        # Check Ollama (skip in dry_run)
        if not dry_run:
            self.llm = OllamaClient()
            if not self.llm.is_running():
                sys.exit("ERROR: Ollama is not running. Start with: ollama serve")
            print(f"✓ Ollama running — model: {OLLAMA_MODEL}")
        else:
            self.llm = None
            print("  [Dry-run mode — LLM evaluation skipped]")

        # Retriever
        print(f"  Initialising vendor retriever…")
        self.retriever = VendorRetriever(vendor_name)

    def evaluate_clause(self, clause: dict) -> dict:
        """
        Evaluate a single clause against vendor documents.
        Returns enriched clause dict with verdict fields added.
        """
        clause_no   = clause["clause_no"]
        requirement = clause["requirement"]
        remarks     = clause.get("remarks", "")

        # Build search query — combine requirement + remarks for richer retrieval
        search_query = requirement
        if remarks and remarks.lower() not in ("nan", ""):
            search_query = f"{requirement} {remarks}"

        # Retrieve evidence
        chunks = self.retriever.search(search_query, n_results=FINAL_TOP_K)
        evidence_block = format_evidence_block(chunks)

        # Check for low-OCR pages
        has_low_ocr = any(
            0 < max(c.get("tess_confidence", 0), c.get("easy_confidence", 0)) < 70
            for c in chunks
        )

        if self.dry_run:
            # Return placeholder verdict in dry-run
            verdict = {
                "status":           "DRY_RUN",
                "explanation":      f"Retrieved {len(chunks)} chunks. LLM skipped.",
                "evidence_pages":   [c.get("page", 0) for c in chunks[:5]],
                "evidence_sources": list({c.get("source_file", "") for c in chunks}),
                "confidence":       "N/A",
                "ocr_warning":      has_low_ocr,
            }
        else:
            # Full LLM evaluation
            verdict = self.llm.evaluate(requirement, evidence_block)
            verdict["ocr_warning"] = verdict.get("ocr_warning", False) or has_low_ocr

        return {
            "clause_no":        clause_no,
            "requirement":      requirement,
            "remarks":          remarks,
            "vendor":           self.vendor_name,
            "status":           verdict.get("status",           STATUS_NOT_FOUND),
            "explanation":      verdict.get("explanation",      ""),
            "evidence_pages":   verdict.get("evidence_pages",   []),
            "evidence_sources": verdict.get("evidence_sources", []),
            "confidence":       verdict.get("confidence",       "LOW"),
            "ocr_warning":      verdict.get("ocr_warning",      False),
            "chunks_retrieved": len(chunks),
        }

    def evaluate_all(self, clauses: list[dict]) -> list[dict]:
        """Evaluate every clause. Returns list of result dicts."""
        results = []
        total   = len(clauses)

        print(f"\nEvaluating {total} clauses for vendor '{self.vendor_name}'…\n")

        for i, clause in enumerate(clauses, 1):
            print(
                f"  [{i:3d}/{total}] Clause {clause['clause_no']:10s} — "
                f"{clause['requirement'][:60]}…",
                end=" ",
                flush=True,
            )

            t0     = time.perf_counter()
            result = self.evaluate_clause(clause)
            elapsed = time.perf_counter() - t0

            status_icon = {
                STATUS_COMPLIANT:     "✓",
                STATUS_NON_COMPLIANT: "✗",
                STATUS_NOT_FOUND:     "?",
            }.get(result["status"], " ")

            print(f"{status_icon} {result['status']:15s}  ({elapsed:.1f}s)")
            results.append(result)

        return results

    def save_results(self, results: list[dict]):
        """Persist raw results JSON for use by the report generator."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{self.vendor_name}_results.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\n  Results saved → {out_path.resolve()}")
        return out_path


# =====================================
# SUMMARY PRINTER
# =====================================
def print_summary(vendor: str, results: list[dict]):
    total       = len(results)
    compliant   = sum(1 for r in results if r["status"] == STATUS_COMPLIANT)
    nc          = sum(1 for r in results if r["status"] == STATUS_NON_COMPLIANT)
    not_found   = sum(1 for r in results if r["status"] == STATUS_NOT_FOUND)
    pct         = (compliant / total * 100) if total else 0

    print(f"\n{'='*50}")
    print(f"  SUMMARY — {vendor}")
    print(f"{'='*50}")
    print(f"  Total clauses     : {total}")
    print(f"  ✓ Compliant       : {compliant}  ({compliant/total*100:.1f}%)")
    print(f"  ✗ Non-Compliant   : {nc}  ({nc/total*100:.1f}%)")
    print(f"  ? Not Found       : {not_found}  ({not_found/total*100:.1f}%)")
    print(f"  Compliance Score  : {pct:.1f}%")
    print(f"{'='*50}\n")


# =====================================
# ENTRY POINT
# =====================================
def discover_vendors() -> list[str]:
    """Auto-discover vendor names from chunks.json vendor field."""
    if not CHUNKS_FILE.exists():
        return []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    vendors = sorted({c.get("vendor", "") for c in chunks if c.get("vendor", "")})
    return vendors


def main():
    parser = argparse.ArgumentParser(
        description="ONGC Compliance Matrix Evaluation Engine"
    )
    parser.add_argument(
        "--vendor", type=str, default=None,
        help="Evaluate a specific vendor (e.g. Vendor_A). Default: all."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Retrieve chunks but skip LLM evaluation."
    )
    parser.add_argument(
        "--matrix", type=str, default=str(MATRIX_FILE),
        help=f"Path to compliance matrix xlsx. Default: {MATRIX_FILE}"
    )
    args = parser.parse_args()

    # Load compliance matrix
    matrix_path = Path(args.matrix)
    clauses     = load_compliance_matrix(matrix_path)

    if not clauses:
        sys.exit("ERROR: No clauses loaded from compliance matrix.")

    # Discover vendors
    if args.vendor:
        vendors = [args.vendor]
    else:
        vendors = discover_vendors()
        if not vendors:
            sys.exit(
                "ERROR: No vendors found in chunks.json.\n"
                "Run run_pipeline.py first to process vendor PDFs."
            )
        print(f"Found vendors: {vendors}")

    # Evaluate each vendor
    all_summaries = {}
    for vendor in vendors:
        engine  = ComplianceEngine(vendor_name=vendor, dry_run=args.dry_run)
        results = engine.evaluate_all(clauses)
        engine.save_results(results)
        print_summary(vendor, results)
        all_summaries[vendor] = results

    print("\nAll vendors evaluated.")
    print(f"Raw results saved in: {RESULTS_DIR.resolve()}")
    print("Run compliance_report_generator.py to generate Excel reports.")


if __name__ == "__main__":
    main()
