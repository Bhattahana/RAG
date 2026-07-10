"""
RAG Chunking Pipeline — connects directly to pdf_ocr_pipeline.py output
========================================================================
Reads:  final_ocr_output.txt  +  ocr_audit.json
Writes: chunks.json  (all chunks with metadata)
        chunks_preview.txt (human readable)
 
Chunking strategy used:
  - Narrative/text pages  → Recursive character splitter with overlap
  - Table regions         → Each table = one chunk (never split tables)
  - Section headings      → Used as chunk boundaries + injected as context
 
Usage:
    python chunker.py
    python chunker.py --txt my_output.txt --json my_audit.json --size 512 --overlap 64
 
Dependencies:
    pip install tiktoken
"""
 
import json
import re
import argparse
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
 
# ── Optional: tiktoken for accurate token counting ────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # Fallback: approximate 1 token ≈ 4 characters
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class ChunkConfig:
    chunk_size: int     = 512    # max tokens per chunk
    chunk_overlap: int  = 64     # overlap tokens between consecutive chunks
    min_chunk_size: int = 50     # discard chunks smaller than this
    add_context_header: bool = True   # prepend doc context to every chunk
    source_name: str    = "tender_document"
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class Chunk:
    chunk_id:        str
    source:          str
    page:            int
    page_type:       str          # TEXT / TABLE / IMAGE / MIXED
    chunk_index:     int          # position within the page
    chunk_type:      str          # "text" | "table"
    section_heading: str          # nearest heading above this chunk
    text:            str          # the actual chunk content
    context_header:  str          # prepended context (for embedding)
    token_count:     int
    char_count:      int
    tesseract_conf:  float
    easyocr_conf:    float
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  HEADING DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
 
# Patterns that typically indicate section headings in tender documents
_HEADING_PATTERNS = [
    re.compile(r'^\s{0,4}(SECTION|CLAUSE|CHAPTER|PART|SCHEDULE|ANNEXURE|APPENDIX)\s+[\dIVXA-Z]', re.IGNORECASE),
    re.compile(r'^\s{0,4}\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{3,60}$'),   # "1.2 Technical Specifications"
    re.compile(r'^\s{0,4}[A-Z][A-Z\s]{4,50}:?\s*$'),                 # "SCOPE OF WORK"
]
 
def detect_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)
 
 
def extract_headings(text: str) -> list[tuple[int, str]]:
    """Return list of (char_offset, heading_text) found in text."""
    headings = []
    pos = 0
    for line in text.splitlines(keepends=True):
        if detect_heading(line):
            headings.append((pos, line.strip()))
        pos += len(line)
    return headings
 
 
def nearest_heading_before(offset: int, headings: list[tuple[int, str]]) -> str:
    """Return the most recent heading that appears before char offset."""
    result = "General"
    for h_offset, h_text in headings:
        if h_offset <= offset:
            result = h_text
        else:
            break
    return result
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  RECURSIVE CHARACTER SPLITTER
# ═══════════════════════════════════════════════════════════════════════════════
 
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # priority order
 
def recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split text recursively by separators until every piece fits max_tokens.
    Then slide an overlap window between consecutive pieces.
    """
    def _split(txt: str, seps: list[str]) -> list[str]:
        if count_tokens(txt) <= max_tokens:
            return [txt]
        if not seps:
            # Force-split by characters as last resort
            mid = len(txt) // 2
            return _split(txt[:mid], seps) + _split(txt[mid:], seps)
 
        sep = seps[0]
        parts = txt.split(sep) if sep else list(txt)
        chunks, current = [], ""
 
        for part in parts:
            candidate = current + (sep if current else "") + part
            if count_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.extend(_split(current, seps[1:]))
                current = part
 
        if current:
            chunks.extend(_split(current, seps[1:]))
        return chunks
 
    raw_chunks = _split(text.strip(), _SEPARATORS)
 
    # Apply overlap: each chunk starts with the tail of the previous one
    if overlap_tokens <= 0 or len(raw_chunks) < 2:
        return [c for c in raw_chunks if c.strip()]
 
    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tokens = _enc.encode(raw_chunks[i - 1]) if 'tiktoken' in dir() else []
        if prev_tokens:
            overlap_text = _enc.decode(prev_tokens[-overlap_tokens:])
        else:
            # Approximate overlap by characters
            approx_chars = overlap_tokens * 4
            overlap_text = raw_chunks[i - 1][-approx_chars:]
        overlapped.append(overlap_text.strip() + "\n" + raw_chunks[i])
 
    return [c for c in overlapped if c.strip()]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE BLOCK EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════
 
_TABLE_RE = re.compile(
    r'\[TABLE_(\d+)_START\](.*?)\[TABLE_\1_END\]',
    re.DOTALL
)
 
def extract_table_blocks(text: str) -> list[tuple[int, int, str, str]]:
    """
    Find all [TABLE_N_START]...[TABLE_N_END] blocks.
    Returns list of (start, end, table_num_str, table_text).
    """
    blocks = []
    for m in _TABLE_RE.finditer(text):
        blocks.append((m.start(), m.end(), m.group(1), m.group(2).strip()))
    return blocks
 
 
def remove_table_blocks(text: str) -> str:
    """Remove all table markers from text, leaving only narrative."""
    return _TABLE_RE.sub("", text).strip()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT HEADER BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
 
def build_context_header(source: str, page: int, section: str, chunk_type: str) -> str:
    """
    Prepend every chunk with a one-line context sentence.
    This dramatically improves retrieval precision.
    """
    return (
        f"[Source: {source} | Page: {page} | "
        f"Section: {section} | Type: {chunk_type.upper()}]\n"
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════
 
class TenderChunker:
 
    def __init__(self, txt_path: str, json_path: str, cfg: ChunkConfig = None):
        self.txt_path  = Path(txt_path)
        self.json_path = Path(json_path)
        self.cfg       = cfg or ChunkConfig()
        self.chunks: list[Chunk] = []
 
        if not self.txt_path.exists():
            raise FileNotFoundError(f"OCR text file not found: {self.txt_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"Audit JSON not found: {self.json_path}")
 
    # ── Public entry ──────────────────────────────────────────────────────────
 
    def run(self) -> list[Chunk]:
        audit = self._load_audit()
        page_conf_map = self._build_conf_map(audit)
        page_texts    = self._parse_txt_by_page()
 
        print(f"\n{'='*60}")
        print(f" TenderChunker — {self.cfg.source_name}")
        print(f" Pages: {len(page_texts)} | chunk_size: {self.cfg.chunk_size} tokens")
        print(f" overlap: {self.cfg.chunk_overlap} tokens | min_size: {self.cfg.min_chunk_size} tokens")
        print(f"{'='*60}\n")
 
        for page_num, raw_text in sorted(page_texts.items()):
            conf = page_conf_map.get(page_num, {})
            page_type = conf.get("page_type", "TEXT")
            tess_conf = conf.get("tesseract_confidence", 0.0)
            easy_conf = conf.get("easyocr_confidence",  0.0)
 
            self._chunk_page(
                page_num=page_num,
                raw_text=raw_text,
                page_type=page_type,
                tess_conf=tess_conf,
                easy_conf=easy_conf,
            )
 
        print(f"\n✅ Total chunks created: {len(self.chunks)}")
        self._print_stats()
        self._save_outputs()
        return self.chunks
 
    # ── Per-page chunking ─────────────────────────────────────────────────────
 
    def _chunk_page(
        self,
        page_num: int,
        raw_text: str,
        page_type: str,
        tess_conf: float,
        easy_conf: float,
    ) -> None:
 
        cfg    = self.cfg
        source = cfg.source_name
 
        # Detect headings across the full page text
        headings = extract_headings(raw_text)
 
        # ── A. Extract and chunk TABLE blocks (never split tables) ─────────────
        table_blocks = extract_table_blocks(raw_text)
 
        for _, _, tbl_num, tbl_text in table_blocks:
            if count_tokens(tbl_text) < cfg.min_chunk_size:
                continue
 
            # Find where this table starts in the original text to get heading
            match = re.search(re.escape(f"[TABLE_{tbl_num}_START]"), raw_text)
            offset = match.start() if match else 0
            section = nearest_heading_before(offset, headings)
 
            ctx = build_context_header(source, page_num, section, "table")
            self._add_chunk(
                source=source, page=page_num, page_type=page_type,
                chunk_type="table", section=section,
                text=tbl_text, context_header=ctx,
                tess_conf=tess_conf, easy_conf=easy_conf,
            )
 
        # ── B. Chunk narrative (non-table) text recursively ───────────────────
        narrative = remove_table_blocks(raw_text)
        if not narrative.strip():
            return
 
        pieces = recursive_split(narrative, cfg.chunk_size, cfg.chunk_overlap)
 
        # Track char offset into narrative for heading lookup
        offset = 0
        for piece in pieces:
            piece = piece.strip()
            if count_tokens(piece) < cfg.min_chunk_size:
                offset += len(piece)
                continue
 
            section = nearest_heading_before(offset, headings)
            ctx = build_context_header(source, page_num, section, "text")
 
            self._add_chunk(
                source=source, page=page_num, page_type=page_type,
                chunk_type="text", section=section,
                text=piece, context_header=ctx,
                tess_conf=tess_conf, easy_conf=easy_conf,
            )
            offset += len(piece)
 
    # ── Chunk factory ─────────────────────────────────────────────────────────
 
    def _add_chunk(
        self,
        source: str, page: int, page_type: str,
        chunk_type: str, section: str,
        text: str, context_header: str,
        tess_conf: float, easy_conf: float,
    ) -> None:
        idx = len([c for c in self.chunks if c.page == page])
        full_text = (context_header + text) if self.cfg.add_context_header else text
 
        chunk = Chunk(
            chunk_id        = str(uuid.uuid4()),
            source          = source,
            page            = page,
            page_type       = page_type,
            chunk_index     = idx,
            chunk_type      = chunk_type,
            section_heading = section,
            text            = text,
            context_header  = context_header,
            token_count     = count_tokens(full_text),
            char_count      = len(full_text),
            tesseract_conf  = tess_conf,
            easyocr_conf    = easy_conf,
        )
        self.chunks.append(chunk)
 
    # ── File parsers ──────────────────────────────────────────────────────────
 
    def _parse_txt_by_page(self) -> dict[int, str]:
        """
        Parse final_ocr_output.txt into {page_number: page_text} dict.
        Looks for the === PAGE N === headers written by the OCR pipeline.
        """
        content = self.txt_path.read_text(encoding="utf-8")
        page_texts: dict[int, str] = {}
 
        # Split on the ======... separator lines that precede each PAGE header
        blocks = re.split(r'={50,}', content)
 
        page_num = None
        for block in blocks:
            block = block.strip()
            if not block:
                continue
 
            # Look for "PAGE N |" header
            m = re.match(r'PAGE\s+(\d+)\s*\|', block)
            if m:
                page_num = int(m.group(1))
                # The actual text follows after the header line(s)
                rest = block[m.end():]
                # Skip the second header line (TESSERACT CONF: ... line)
                lines = rest.splitlines()
                # Drop lines that are pure metadata
                text_lines = [
                    l for l in lines
                    if not re.match(r'^\s*(TESSERACT|EASYOCR|TYPE:|TABLES:|IMAGES:)', l, re.I)
                ]
                page_texts[page_num] = "\n".join(text_lines).strip()
            elif page_num is not None and block:
                # Continuation block for same page (shouldn't happen, but safe)
                page_texts[page_num] = page_texts.get(page_num, "") + "\n" + block
 
        return page_texts
 
    def _load_audit(self) -> dict:
        return json.loads(self.json_path.read_text(encoding="utf-8"))
 
    def _build_conf_map(self, audit: dict) -> dict[int, dict]:
        result = {}
        for p in audit.get("pages", []):
            result[p["page"]] = {
                "page_type":            p.get("page_type", "TEXT"),
                "tesseract_confidence": p.get("tesseract_confidence", 0.0),
                "easyocr_confidence":   p.get("easyocr_confidence", 0.0),
            }
        return result
 
    # ── Output writers ────────────────────────────────────────────────────────
 
    def _save_outputs(self) -> None:
        # ── chunks.json ──────────────────────────────────────────────────────
        out_json = Path("chunks.json")
        payload = {
            "source":       self.cfg.source_name,
            "total_chunks": len(self.chunks),
            "chunk_size":   self.cfg.chunk_size,
            "overlap":      self.cfg.chunk_overlap,
            "chunks":       [asdict(c) for c in self.chunks],
        }
        out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n📄 chunks.json saved → {out_json.resolve()}")
 
        # ── chunks_preview.txt ───────────────────────────────────────────────
        out_txt = Path("chunks_preview.txt")
        lines = []
        for c in self.chunks:
            lines.append(f"{'─'*60}")
            lines.append(
                f"CHUNK {c.chunk_id[:8]}  |  Page {c.page}  |  "
                f"Type: {c.chunk_type.upper()}  |  Section: {c.section_heading}"
            )
            lines.append(f"Tokens: {c.token_count}  |  Chars: {c.char_count}  |  "
                         f"OCR conf: tess={c.tesseract_conf}% easy={c.easyocr_conf}%")
            lines.append("")
            lines.append(c.context_header.strip())
            lines.append(c.text[:400] + ("…" if len(c.text) > 400 else ""))
            lines.append("")
 
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"📄 chunks_preview.txt saved → {out_txt.resolve()}")
 
    # ── Stats ─────────────────────────────────────────────────────────────────
 
    def _print_stats(self) -> None:
        if not self.chunks:
            return
        tokens = [c.token_count for c in self.chunks]
        text_c  = sum(1 for c in self.chunks if c.chunk_type == "text")
        table_c = sum(1 for c in self.chunks if c.chunk_type == "table")
 
        print(f"\n{'─'*40}")
        print(f"  Text chunks  : {text_c}")
        print(f"  Table chunks : {table_c}")
        print(f"  Avg tokens   : {sum(tokens)//len(tokens)}")
        print(f"  Min tokens   : {min(tokens)}")
        print(f"  Max tokens   : {max(tokens)}")
        print(f"{'─'*40}\n")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
 
def parse_args():
    p = argparse.ArgumentParser(description="Chunk OCR output for RAG pipeline")
    p.add_argument("--txt",     default="final_ocr_output.txt", help="OCR text file")
    p.add_argument("--json",    default="ocr_audit.json",       help="OCR audit JSON")
    p.add_argument("--size",    type=int, default=512,          help="Max tokens per chunk")
    p.add_argument("--overlap", type=int, default=64,           help="Overlap tokens")
    p.add_argument("--source",  default="tender_document",      help="Document name/ID")
    return p.parse_args()
 
import re

MIN_CHUNK_CHARS = 100

TABLE_BLOCK_RE = re.compile(
    r"\[TABLE_(\d+)_START\](.*?)\[TABLE_\1_END\]",
    re.DOTALL
)

def clean_ocr(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_pages(raw: str):
    pages = []

    blocks = re.split(r"={50,}", raw)

    current_page = None
    current_type = "TEXT"
    content = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        m = re.search(r"PAGE\s+(\d+)\s+\|\s+TYPE:\s*([A-Z]+)", block)

        if m:
            if current_page is not None:
                pages.append({
                    "page": current_page,
                    "page_type": current_type,
                    "content": "\n".join(content).strip()
                })

            current_page = int(m.group(1))
            current_type = m.group(2)

            lines = block.splitlines()
            content = lines[2:] if len(lines) > 2 else []

        else:
            content.extend(block.splitlines())

    if current_page is not None:
        pages.append({
            "page": current_page,
            "page_type": current_type,
            "content": "\n".join(content).strip()
        })

    return pages

def merge_cross_page(pages):
    return pages

def get_sem_model():
    return None

def extract_sections(text):
    current_header = ""
    buffer = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.isupper() and len(line.split()) <= 8 and len(line) < 80:
            if buffer:
                yield current_header, "\n".join(buffer)
                buffer = []
            current_header = line
        else:
            buffer.append(line)

    if buffer:
        yield current_header, "\n".join(buffer)

def table_to_nl(table_text, table_num):
    rows = [r.strip() for r in table_text.splitlines() if r.strip()]
    return [f"Table {table_num}: {row}" for row in rows]

def fallback_split(text, max_chars=1500):
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    return [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
    ]

def semantic_split(text, model=None):
    return fallback_split(text)

if __name__ == "__main__":
    args = parse_args()
 
    cfg = ChunkConfig(
        chunk_size    = args.size,
        chunk_overlap = args.overlap,
        source_name   = args.source,
    )
 
    chunker = TenderChunker(
        txt_path  = args.txt,
        json_path = args.json,
        cfg       = cfg,
    )
    chunker.run()
