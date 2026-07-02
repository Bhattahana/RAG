

import json
import os
import sys
import textwrap
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

try:
    import requests
except ImportError:
    sys.exit("Run:  pip install requests")

try:
    from retriever import HybridRetriever
except ImportError:
    sys.exit("retriever.py not found — place chatbot.py in the ONGC_RAG folder.")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these two values
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.A")
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:3b"      # change to qwen2.5:7b for better answers

RETRIEVAL_TOP_K = 10
HISTORY_TURNS   = 6
FREE_RPM_LIMIT  = 14                # stay under Gemini's 15 RPM free limit

HISTORY_DIR     = Path("chat_history")
RESULTS_DIR     = Path("compliance_results")
VENDORS_DIR     = Path("vendors")   # ONGC_RAG/vendors/<Vendor_Name>/ — used to auto-detect vendors

SYSTEM_PROMPT = """You are a technical assistant specialising in oil and gas equipment \
specifications and tender documents for ONGC (Oil and Natural Gas Corporation).

Answer questions STRICTLY based on the retrieved specification context provided.

Rules:
1. Only use information from the provided context. Never hallucinate specifications.
2. Always reference your source inline: (Page X, <section name>, <filename>).
3. If a value is not in the context say:
   "Not found in retrieved sections — try rephrasing or check the original document."
4. For numerical specs (power, voltage, speed, pressure) quote exact values with units.
5. If any chunk has low OCR confidence (<70%), warn: ⚠ Verify against original PDF.
6. Use bullet points for lists of specifications.
7. Mark compliance clearly: ✓ Compliant / ✗ Non-compliant / ? Not specified.
8. Keep answers concise and structured."""

COMPARISON_SYSTEM_PROMPT = """You are comparing how multiple vendors responded to the SAME \
tender question, based on per-vendor answers that were already grounded in each vendor's own \
documents (given below).

Rules:
1. Do not invent any data — only use what each vendor's answer states.
2. Produce a markdown table with columns: Vendor | Key Spec / Answer | Compliance.
3. After the table, add a short "Differences" note calling out any mismatch, gap, or missing data.
4. If one vendor has "Not found" and another has data, say so explicitly — don't guess.
5. End with one line: either a recommendation on which vendor better meets the requirement, or a \
note that more information is needed to decide."""


# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER  (Gemini free tier — 15 RPM ceiling)
# ══════════════════════════════════════════════════════════════════════════════
class _RateLimiter:
    def __init__(self, rpm: int = FREE_RPM_LIMIT):
        self.rpm      = rpm
        self.window   = 60.0
        self.requests = deque()

    def wait(self):
        now = time.monotonic()
        while self.requests and now - self.requests[0] > self.window:
            self.requests.popleft()
        if len(self.requests) >= self.rpm:
            sleep_for = self.window - (now - self.requests[0]) + 0.2
            if sleep_for > 0:
                print(f"\n  ⏳ Rate limit — waiting {sleep_for:.0f}s…", flush=True)
                time.sleep(sleep_for)
        self.requests.append(time.monotonic())

_gemini_limiter = _RateLimiter()


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class LLMBackend:
    """Abstract interface — both backends implement stream() and name."""
    backend_name: str = "base"

    def stream(
        self,
        messages:    list[dict],
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 2000,
    ) -> Iterator[str]:
        raise NotImplementedError

    def usage(self) -> str:
        return ""

    def is_available(self) -> bool:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND 1 — GEMINI (free cloud API)
# ══════════════════════════════════════════════════════════════════════════════
class GeminiBackend(LLMBackend):
    backend_name = "Gemini"

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        self.api_key         = api_key
        self.model           = model
        self.session         = requests.Session()
        self._prompt_tokens  = 0
        self._output_tokens  = 0
        self._call_count     = 0

    def is_available(self) -> bool:
        """Test the key with a tiny probe request."""
        if not self.api_key or self.api_key == "YOUR_GEMINI_KEY_HERE":
            return False
        try:
            url = (
                f"{GEMINI_BASE_URL}/models/{self.model}:generateContent"
                f"?key={self.api_key}"
            )
            payload = {
                "contents":        [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            }
            r = self.session.post(url, json=payload, timeout=8)
            return r.status_code == 200
        except Exception:
            return False

    def _url(self, stream: bool = False) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        suffix = "&alt=sse" if stream else ""
        return f"{GEMINI_BASE_URL}/models/{self.model}:{action}?key={self.api_key}{suffix}"

    def _to_contents(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            out.append({"role": role, "parts": [{"text": m["content"]}]})
        return out

    def _post(
        self,
        messages: list[dict], system: str,
        temperature: float, max_tokens: int, stream: bool,
    ) -> requests.Response:
        payload = {
            "contents":         self._to_contents(messages),
            "generationConfig": {
                "temperature":     temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        _gemini_limiter.wait()
        for attempt in range(1, 4):
            try:
                resp = self.session.post(
                    self._url(stream), json=payload,
                    stream=stream, timeout=120,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    print(f"\n  ⏳ 429 — sleeping {wait}s…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                self._call_count += 1
                return resp
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    raise
                time.sleep(5 * attempt)
        raise RuntimeError("Gemini: max retries exceeded")

    def stream(
        self,
        messages:    list[dict],
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 2000,
    ) -> Iterator[str]:
        resp = self._post(messages, system, temperature, max_tokens, stream=True)
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
                if "usageMetadata" in chunk:
                    m = chunk["usageMetadata"]
                    self._prompt_tokens += m.get("promptTokenCount",    0)
                    self._output_tokens += m.get("candidatesTokenCount", 0)
                parts = (
                    chunk.get("candidates", [{}])[0]
                         .get("content", {})
                         .get("parts", [])
                )
                for part in parts:
                    tok = part.get("text", "")
                    if tok:
                        yield tok
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def usage(self) -> str:
        total = self._prompt_tokens + self._output_tokens
        return (
            f"Gemini | calls:{self._call_count} | "
            f"tokens:{self._prompt_tokens}↑{self._output_tokens}↓({total}) | "
            f"cost:$0.00"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND 2 — OLLAMA (local, free, offline)
# ══════════════════════════════════════════════════════════════════════════════
class OllamaBackend(LLMBackend):
    backend_name = "Ollama"

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model:    str = OLLAMA_MODEL,
    ):
        self.base_url       = base_url.rstrip("/")
        self.model          = model
        self._call_count    = 0
        self._total_tokens  = 0

    def is_available(self) -> bool:
        """Check Ollama server is running AND the model is pulled."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=4)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Accept partial match: "qwen2.5:3b" matches "qwen2.5:3b-instruct-..."
            return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    def _build_messages(self, messages: list[dict], system: str) -> list[dict]:
        """Prepend system message in Ollama format."""
        out = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def stream(
        self,
        messages:    list[dict],
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 2000,
    ) -> Iterator[str]:
        payload = {
            "model":    self.model,
            "messages": self._build_messages(messages, system),
            "stream":   True,
            "options": {
                "temperature": temperature,
                "top_p":       0.9,
                "num_ctx":     8192,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload, stream=True, timeout=120,
        )
        resp.raise_for_status()
        self._call_count += 1

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    self._total_tokens += 1
                    yield token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue

    def usage(self) -> str:
        return (
            f"Ollama ({self.model}) | calls:{self._call_count} | "
            f"~tokens:{self._total_tokens} | cost:$0.00"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND SELECTOR  — runs at startup
# ══════════════════════════════════════════════════════════════════════════════
def select_backend(force: Optional[str] = None) -> LLMBackend:
    """
    Determine which backend to use.

    Priority:
      1. If force="gemini"  → Gemini only (error if unavailable)
      2. If force="ollama"  → Ollama only (error if unavailable)
      3. If force=None      → show menu, auto-detect availability,
                              fall back automatically if one is down
    """
    gemini = GeminiBackend()
    ollama = OllamaBackend()

    # ── Forced mode ───────────────────────────────────────────────────────────
    if force == "gemini":
        print("  Checking Gemini…", end=" ", flush=True)
        if gemini.is_available():
            print("✓ online")
            return gemini
        else:
            sys.exit(
                "\n  ✗ Gemini unavailable.\n"
                "  Check GEMINI_API_KEY in chatbot.py and internet connection."
            )

    if force == "ollama":
        print(f"  Checking Ollama ({OLLAMA_MODEL})…", end=" ", flush=True)
        if ollama.is_available():
            print("✓ running")
            return ollama
        else:
            sys.exit(
                f"\n  ✗ Ollama unavailable.\n"
                f"  Make sure Ollama is running:  ollama serve\n"
                f"  And the model is pulled:      ollama pull {OLLAMA_MODEL}"
            )

    # ── Interactive choice ────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  ONGC Chatbot — Choose LLM Backend")
    print("═"*58)

    # Probe both silently
    print("  Detecting available backends…")

    gemini_ok = gemini.is_available()
    ollama_ok = ollama.is_available()

    gemini_status = "✓ available (free, cloud)" if gemini_ok else "✗ not available"
    ollama_status = (
        f"✓ available ({OLLAMA_MODEL}, local)"
        if ollama_ok
        else f"✗ not running  (ollama serve + ollama pull {OLLAMA_MODEL})"
    )

    # Key missing hint
    if not gemini_ok and GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        gemini_status += "  ← set GEMINI_API_KEY in chatbot.py"

    print(f"\n  [1] Gemini 1.5 Flash  {gemini_status}")
    print(f"  [2] Ollama local      {ollama_status}")
    print(f"  [a] Auto (best available)")
    print()

    # If neither is available, exit cleanly
    if not gemini_ok and not ollama_ok:
        print("  ✗ Neither backend is available.")
        print("  Option A: Set GEMINI_API_KEY in chatbot.py (free key from aistudio.google.com)")
        print(f"  Option B: Run  ollama serve  and  ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    # If only one is available, auto-select it
    if gemini_ok and not ollama_ok:
        print("  → Only Gemini is available. Using Gemini automatically.")
        return gemini
    if ollama_ok and not gemini_ok:
        print(f"  → Only Ollama is available. Using Ollama ({OLLAMA_MODEL}) automatically.")
        return ollama

    # Both available — ask user
    while True:
        try:
            choice = input("  Your choice [1/2/a]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Defaulting to Gemini.")
            return gemini

        if choice in ("1", "gemini", "g"):
            print(f"  ✓ Using Gemini ({GEMINI_MODEL})")
            return gemini
        elif choice in ("2", "ollama", "o"):
            print(f"  ✓ Using Ollama ({OLLAMA_MODEL})")
            return ollama
        elif choice in ("a", "auto", ""):
            # Prefer Gemini in auto mode (better quality, free)
            print(f"  ✓ Auto-selected Gemini ({GEMINI_MODEL})")
            return gemini
        else:
            print("  Please enter 1, 2, or a")


# ══════════════════════════════════════════════════════════════════════════════
#  VENDOR DETECTION + SELECTION  — runs at startup
# ══════════════════════════════════════════════════════════════════════════════
def discover_vendors() -> list[str]:
    """
    Auto-detect vendors from sub-folders of VENDORS_DIR, e.g.:
        ONGC_RAG/vendors/Vendor_A
        ONGC_RAG/vendors/Vendor_B
    Falls back to an empty list if the folder doesn't exist (vendor metadata
    on chunks is still used for filtering either way).
    """
    if not VENDORS_DIR.exists():
        return []
    return sorted(p.name for p in VENDORS_DIR.iterdir() if p.is_dir())


def select_vendor_mode(
    vendors:       list[str],
    force_vendor:  Optional[str] = None,
    force_compare: bool          = False,
) -> tuple[Optional[str], list[str]]:
    """
    Decide the vendor scope for the session.

    Returns (single_vendor, compare_list):
      • single_vendor set, compare_list empty  → normal single-vendor filter
      • single_vendor None, compare_list set   → side-by-side comparison mode
      • both empty/None                        → no filter, search all vendors
    """
    if not vendors:
        if force_vendor:
            return force_vendor, []
        return None, []

    if force_compare:
        print(f"  ✓ Comparing all detected vendors: {', '.join(vendors)}")
        return None, vendors

    if force_vendor:
        match = next((v for v in vendors if v.lower() == force_vendor.lower()), force_vendor)
        print(f"  ✓ Vendor filter → {match}")
        return match, []

    print("\n" + "═"*58)
    print("  Select Vendor Scope")
    print("═"*58)
    for i, v in enumerate(vendors, 1):
        print(f"  [{i}] {v}")
    print(f"  [c] Compare ALL vendors ({' vs '.join(vendors)})")
    print(f"  [n] No filter — search across all vendors normally")
    print()

    while True:
        try:
            choice = input("  Your choice [1/2/.../c/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Defaulting to no filter.")
            return None, []

        if choice in ("n", "none", ""):
            print("  ✓ No vendor filter — searching across all vendors")
            return None, []
        if choice in ("c", "compare"):
            print(f"  ✓ Comparing: {' vs '.join(vendors)}")
            return None, vendors
        if choice.isdigit() and 1 <= int(choice) <= len(vendors):
            picked = vendors[int(choice) - 1]
            print(f"  ✓ Vendor filter → {picked}")
            return picked, []
        match = next((v for v in vendors if v.lower() == choice), None)
        if match:
            print(f"  ✓ Vendor filter → {match}")
            return match, []
        print("  Please enter a number, 'c' to compare, or 'n' for no filter.")


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════
class ConversationMemory:
    def __init__(self, max_turns: int = HISTORY_TURNS):
        self.max_turns = max_turns
        self.history:  list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get(self) -> list[dict]:
        return list(self.history)

    def clear(self):
        self.history = []

    def turn_count(self) -> int:
        return len(self.history) // 2


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HISTORY  (full record + feedback)
# ══════════════════════════════════════════════════════════════════════════════
class SessionHistory:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turns:     list[dict] = []

    def record(
        self,
        question:   str,
        answer:     str,
        chunks:     list[dict],
        feedback:   str = "pending",
        correction: str = "",
        clause_no:  str = "",
    ):
        self.turns.append({
            "turn":       len(self.turns) + 1,
            "timestamp":  datetime.now().isoformat(),
            "clause_no":  clause_no,
            "question":   question,
            "answer":     answer,
            "sources":    _summarise_sources(chunks),
            "feedback":   feedback,
            "correction": correction,
        })

    def update_last_feedback(self, feedback: str, correction: str = ""):
        if self.turns:
            self.turns[-1]["feedback"]   = feedback
            self.turns[-1]["correction"] = correction

    def save(self) -> Path:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = HISTORY_DIR / f"{self.session_id}_session.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id":  self.session_id,
                "total_turns": len(self.turns),
                "turns":       self.turns,
            }, f, indent=4, ensure_ascii=False)
        return path

    def accuracy_stats(self) -> dict:
        rated   = [t for t in self.turns if t["feedback"] in ("correct", "wrong")]
        correct = sum(1 for t in rated if t["feedback"] == "correct")
        return {
            "total":    len(self.turns),
            "rated":    len(rated),
            "correct":  correct,
            "wrong":    len(rated) - correct,
            "skipped":  sum(1 for t in self.turns if t["feedback"] == "skip"),
            "pending":  sum(1 for t in self.turns if t["feedback"] == "pending"),
            "accuracy": f"{correct/len(rated)*100:.1f}%" if rated else "N/A",
        }


def _summarise_sources(chunks: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for c in chunks:
        key = (c.get("source_file", ""), c.get("page", 0), c.get("section", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    c.get("source_file", "unknown"),
                "vendor":  c.get("vendor", ""),
                "page":    c.get("page", "?"),
                "section": c.get("section", "") or "General",
                "score":   round(c.get("score", 0), 4),
            })
    return sources


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE REFERENCE BLOCK
# ══════════════════════════════════════════════════════════════════════════════
def build_reference_block(chunks: list[dict]) -> str:
    if not chunks:
        return "  (no sources retrieved)"

    lines = ["\n  ┌─ Sources ────────────────────────────────────────────────────"]
    seen  = set()
    rank  = 1

    for c in chunks:
        page    = c.get("page",        "?")
        section = c.get("section",     "") or "General"
        source  = c.get("source_file", "unknown")
        vendor  = c.get("vendor",      "")
        ctype   = c.get("chunk_type",  "text")
        score   = c.get("score",       0)
        tconf   = c.get("tess_confidence", 0)
        econf   = c.get("easy_confidence", 0)
        ocr_c   = max(tconf, econf)

        key = (source, page, section)
        if key in seen:
            continue
        seen.add(key)

        icon    = "📊" if "table" in ctype else "📝"
        v_tag   = f"[{vendor}] " if vendor else ""
        ocr_tag = (
            f" ⚠ OCR:{ocr_c:.0f}%" if 0 < ocr_c < 70
            else (f" OCR:{ocr_c:.0f}%" if ocr_c else "")
        )
        bar = "█" * int(min(score, 1.0) * 10) + "░" * (10 - int(min(score, 1.0) * 10))

        lines.append(f"  │  [{rank}] {icon} {v_tag}{source}")
        lines.append(f"  │      Page {page}  ·  {section}")
        lines.append(f"  │      Relevance {bar} {score:.3f}{ocr_tag}")
        rank += 1

    lines.append("  └──────────────────────────────────────────────────────────")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT BLOCK  (fed into LLM prompt)
# ══════════════════════════════════════════════════════════════════════════════
def build_context_block(chunks: list[dict]) -> str:
    lines = ["=== RETRIEVED SPECIFICATION CONTEXT ===\n"]
    for i, c in enumerate(chunks, 1):
        page     = c.get("page",           "?")
        section  = c.get("section",        "") or "General"
        source   = c.get("source_file",    "unknown")
        vendor   = c.get("vendor",         "")
        ctype    = c.get("chunk_type",     "text")
        tconf    = c.get("tess_confidence", 0)
        econf    = c.get("easy_confidence", 0)
        conf     = max(tconf, econf)
        score    = c.get("score",          0)
        text     = c.get("text",           "").strip()

        v_tag    = f"[{vendor}] " if vendor else ""
        ocr_note = f"⚠ OCR:{conf:.0f}%" if 0 < conf < 70 else f"OCR:{conf:.0f}%"

        lines.append(
            f"[{i}] {v_tag}{source} | Page {page} | {section} | "
            f"{ctype} | {ocr_note} | relevance:{score:.4f}"
        )
        lines.append(text)
        lines.append("")

    lines.append("=== END OF CONTEXT ===")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def ask_feedback(session: SessionHistory) -> tuple[str, str]:
    print("\n  Was this answer correct?")
    print("  [y] Yes   [n] No — I'll correct it   [s] Skip")

    while True:
        try:
            ans = input("  → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "skip", ""

        if ans in ("y", "yes"):
            session.update_last_feedback("correct")
            print("  ✓ Marked as correct")
            return "correct", ""

        elif ans in ("n", "no"):
            print("  Enter the correct answer / what was wrong:")
            try:
                correction = input("  Correction: ").strip()
            except (EOFError, KeyboardInterrupt):
                correction = ""
            session.update_last_feedback("wrong", correction)
            print("  ✓ Correction saved")
            return "wrong", correction

        elif ans in ("s", "skip", ""):
            session.update_last_feedback("skip")
            return "skip", ""

        else:
            print("  Please enter y, n, or s")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPLIANCE REPORT  (from session history)
# ══════════════════════════════════════════════════════════════════════════════
def generate_session_report(session: SessionHistory, vendor: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for t in session.turns:
        status = {
            "correct": "COMPLIANT",
            "wrong":   "NON_COMPLIANT",
        }.get(t["feedback"], "NOT_FOUND")

        results.append({
            "clause_no":        t.get("clause_no") or f"Q{t['turn']}",
            "requirement":      t["question"],
            "vendor":           vendor or "session",
            "status":           status,
            "explanation":      t["answer"][:300],
            "evidence_pages":   [s["page"] for s in t["sources"]],
            "evidence_sources": list({s["file"] for s in t["sources"]}),
            "confidence":       {"correct": "HIGH", "wrong": "LOW"}.get(
                                    t["feedback"], "MEDIUM"),
            "ocr_warning":      any("⚠" in s.get("section", "") for s in t["sources"]),
            "user_feedback":    t["feedback"],
            "user_correction":  t.get("correction", ""),
        })

    fname = f"{session.session_id}_{vendor or 'session'}_manual_report.json"
    path  = RESULTS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return path


def print_session_report(session: SessionHistory):
    stats = session.accuracy_stats()
    turns = session.turns

    print(f"\n{'═'*58}")
    print(f"  SESSION COMPLIANCE REPORT")
    print(f"{'═'*58}")
    print(f"  Session ID     : {session.session_id}")
    print(f"  Total Q&A turns: {stats['total']}")
    print(f"  Rated          : {stats['rated']}")
    print(f"  ✓ Correct      : {stats['correct']}")
    print(f"  ✗ Wrong        : {stats['wrong']}")
    print(f"  — Skipped      : {stats['skipped']}")
    print(f"  ○ Pending      : {stats['pending']}")
    print(f"  Answer Accuracy: {stats['accuracy']}")
    print(f"{'─'*58}")

    if turns:
        print("  Turn-by-turn summary:")
        for t in turns:
            icon   = {"correct":"✓","wrong":"✗","skip":"—","pending":"○"}.get(t["feedback"],"○")
            clause = f"[{t['clause_no']}] " if t.get("clause_no") else ""
            print(f"  {icon} Turn {t['turn']:2d} {clause}{t['question'][:55]}…")
            if t.get("correction"):
                print(f"       ✏ {t['correction'][:70]}")

    print(f"{'═'*58}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
class TenderChatbot:

    def __init__(self, backend: LLMBackend):
        self.llm           = backend
        print(f"\n  Initialising retriever…")
        self.retriever     = HybridRetriever(use_reranker=False)
        self.memory        = ConversationMemory(max_turns=HISTORY_TURNS)
        self.session       = SessionHistory()
        self.last_chunks:  list[dict]    = []
        self.last_answer:  str           = ""
        self.strict_mode:  bool          = False
        self.active_vendor: Optional[str] = None
        self.compare_vendors: list[str]   = []
        self.feedback_mode: bool         = True

    # ── vendor filter ─────────────────────────────────────────────────────────
    def _set_vendor(self, name: str):
        self.active_vendor = name.strip() if name.strip() else None
        self.compare_vendors = []
        if self.active_vendor:
            print(f"  ✓ Vendor filter → {self.active_vendor}")
        else:
            print("  ✓ Vendor filter cleared")
        self.memory.clear()

    # ── compare mode ──────────────────────────────────────────────────────────
    def _set_compare(self, vendors: list[str]):
        vendors = [v.strip() for v in vendors if v.strip()]
        if len(vendors) < 2:
            print("  ✗ Need at least 2 vendors to compare.")
            return
        self.compare_vendors = vendors
        self.active_vendor   = None
        print(f"  ✓ Compare mode → {' vs '.join(vendors)}")
        self.memory.clear()

    # ── switch vendor scope mid-session ────────────────────────────────────────
    def _switch_scope(self):
        """Interactive vendor-scope switcher — same picker shown at startup, mirrors /backend."""
        vendors = discover_vendors()
        if not vendors:
            print("  No vendor folders detected under 'vendors/' — nothing to switch.")
            return

        current = (
            f"Comparing {' vs '.join(self.compare_vendors)}" if self.compare_vendors
            else (self.active_vendor or "All vendors")
        )
        print(f"\n  Current scope: {current}")
        single, compare_list = select_vendor_mode(vendors)
        if compare_list:
            self._set_compare(compare_list)
        elif single:
            self._set_vendor(single)
        else:
            self._set_vendor("")

    def _quick_scope_switch(self, text: str) -> bool:
        """
        Fast scope switching without the /vendor or /compare syntax — e.g. just
        typing "Vendor_A", "compare", "compare all", or "all vendors" on its own.
        Only fires on an exact match against a detected vendor name or a small
        fixed phrase list, so it never hijacks a real question. Returns True if
        the message was handled as a scope switch.
        """
        lo      = text.strip().lower()
        vendors = discover_vendors()

        if vendors:
            match = next((v for v in vendors if v.lower() == lo), None)
            if match:
                self._set_vendor(match)
                return True

        if lo in ("compare", "compare all", "compare both", "compare vendors"):
            if len(vendors) >= 2:
                self._set_compare(vendors)
            else:
                print("  Need at least 2 detected vendors under 'vendors/' to compare.")
            return True

        if lo in ("all vendors", "no filter", "clear filter", "clear vendor filter"):
            self._set_vendor("")
            return True

        return False

    # ── switch backend mid-session ────────────────────────────────────────────
    def _switch_backend(self):
        """Let the user swap backend without restarting."""
        print(f"\n  Current backend: {self.llm.backend_name}")
        print("  [1] Gemini  [2] Ollama  [c] Cancel")
        try:
            ch = input("  → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if ch in ("1", "gemini", "g"):
            g = GeminiBackend()
            if g.is_available():
                self.llm = g
                print(f"  ✓ Switched to Gemini ({GEMINI_MODEL})")
            else:
                print("  ✗ Gemini not available — keeping current backend")
        elif ch in ("2", "ollama", "o"):
            o = OllamaBackend()
            if o.is_available():
                self.llm = o
                print(f"  ✓ Switched to Ollama ({OLLAMA_MODEL})")
            else:
                print(f"  ✗ Ollama not available — keeping current backend")
        else:
            print("  Cancelled")

    # ── core ask ──────────────────────────────────────────────────────────────
    def ask(self, question: str, clause_no: str = "") -> str:
        if self.compare_vendors:
            return self.ask_compare(question, clause_no=clause_no)

        # 1. Retrieve
        chunks = self.retriever.search(question, n_results=RETRIEVAL_TOP_K)

        # Client-side vendor filter (works even if ChromaDB metadata missing)
        if self.active_vendor:
            filtered = [
                c for c in chunks
                if c.get("vendor", "").lower() == self.active_vendor.lower()
            ]
            chunks = filtered if filtered else chunks

        self.last_chunks = chunks

        if not chunks:
            answer = (
                "No relevant content found in the indexed documents. "
                "Try rephrasing or ensure vendor documents have been processed."
            )
            print(f"\n  🤖 {answer}")
            self.session.record(question, answer, [], clause_no=clause_no)
            return answer

        # 2. Build context + messages
        context = build_context_block(chunks)
        system  = SYSTEM_PROMPT
        if self.active_vendor:
            system += (
                f"\n\nACTIVE VENDOR: {self.active_vendor}. "
                "Only cite evidence from this vendor's documents."
            )

        messages = list(self.memory.get()) + [{
            "role":    "user",
            "content": f"{context}\n\nQuestion: {question}",
        }]

        # 3. Stream answer
        backend_tag = f"[{self.llm.backend_name}]"
        vendor_tag  = f"[{self.active_vendor}] " if self.active_vendor else ""
        print(f"\n  🤖 {backend_tag} {vendor_tag}Answer:\n")

        answer = ""
        for token in self.llm.stream(
            messages    = messages,
            system      = system,
            temperature = 0.1,
            max_tokens  = 2000,
        ):
            print(token, end="", flush=True)
            answer += token
        print("\n")

        # 4. Update memory + session
        self.memory.add("user",      question)
        self.memory.add("assistant", answer)
        self.last_answer = answer
        self.session.record(question, answer, chunks, clause_no=clause_no)

        return answer

    # ── compare ask (side-by-side, multi-vendor) ────────────────────────────────
    def ask_compare(self, question: str, clause_no: str = "") -> str:
        """
        Retrieves + answers per-vendor independently (so each answer is grounded
        only in that vendor's own documents), then asks the LLM once more to
        synthesise a side-by-side comparison table across vendors.
        """
        vendors = self.compare_vendors
        print(f"\n  🔍 Comparing {len(vendors)} vendor(s): {', '.join(vendors)}")

        per_vendor_answers: dict[str, str]        = {}
        per_vendor_chunks:  dict[str, list[dict]]  = {}

        for vendor in vendors:
            # Over-fetch so filtering down to one vendor still leaves enough context
            raw_chunks = self.retriever.search(question, n_results=RETRIEVAL_TOP_K * 2)
            v_chunks   = [
                c for c in raw_chunks
                if c.get("vendor", "").lower() == vendor.lower()
            ]
            per_vendor_chunks[vendor] = v_chunks

            print(f"\n  ── {vendor} " + "─"*max(2, 50 - len(vendor)))

            if not v_chunks:
                answer = "No relevant content found in this vendor's indexed documents."
                print(f"  {answer}")
                per_vendor_answers[vendor] = answer
                continue

            context = build_context_block(v_chunks)
            system  = SYSTEM_PROMPT + (
                f"\n\nACTIVE VENDOR: {vendor}. Only cite evidence from this vendor's documents."
            )
            messages = [{
                "role":    "user",
                "content": f"{context}\n\nQuestion: {question}",
            }]

            answer = ""
            for token in self.llm.stream(
                messages    = messages,
                system      = system,
                temperature = 0.1,
                max_tokens  = 1200,
            ):
                print(token, end="", flush=True)
                answer += token
            print()
            per_vendor_answers[vendor] = answer

        # Synthesis pass — ask the LLM to diff the per-vendor answers
        print(f"\n  ── Comparison Summary " + "─"*36 + "\n")
        synth_body = "\n\n".join(
            f"### {v}'s answer:\n{a}" for v, a in per_vendor_answers.items()
        )
        synth_messages = [{
            "role":    "user",
            "content": f"Original question: {question}\n\n{synth_body}",
        }]

        synthesis = ""
        for token in self.llm.stream(
            messages    = synth_messages,
            system      = COMPARISON_SYSTEM_PROMPT,
            temperature = 0.1,
            max_tokens  = 800,
        ):
            print(token, end="", flush=True)
            synthesis += token
        print("\n")

        # Bookkeeping — combined chunks across vendors for /sources, session record etc.
        combined_chunks  = [c for cl in per_vendor_chunks.values() for c in cl]
        self.last_chunks = combined_chunks

        final_record = (
            "\n\n".join(f"[{v}]\n{a}" for v, a in per_vendor_answers.items())
            + f"\n\n[Comparison]\n{synthesis}"
        )
        self.last_answer = final_record
        self.session.record(question, final_record, combined_chunks, clause_no=clause_no)
        return final_record

    # ── CLI loop ──────────────────────────────────────────────────────────────
    def run_cli(self):
        print("\n" + "═"*60)
        print("  ONGC Tender Chatbot")
        print(f"  Backend : {self.llm.backend_name}"
              + (f" ({GEMINI_MODEL})" if isinstance(self.llm, GeminiBackend)
                 else f" ({OLLAMA_MODEL})"))
        if self.compare_vendors:
            print(f"  Scope   : Comparing {' vs '.join(self.compare_vendors)}")
        else:
            print(f"  Scope   : {self.active_vendor or 'All vendors'}")
        print(f"  Feedback: {'ON' if self.feedback_mode else 'OFF'}")
        print("═"*60)
        print("  /vendor  /compare  /scope  /backend  /compliance  /report  /history")
        print("  /sources  /feedback  /clear  /mode  /help  /quit")
        print("═"*60 + "\n")

        while True:
            try:
                scope  = " vs ".join(self.compare_vendors) if self.compare_vendors else (self.active_vendor or "All")
                prompt = f"  [{self.llm.backend_name}|{scope}] You: "
                user   = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

            if not user:
                continue

            lo = user.lower()

            if lo == "/quit":
                self._exit()
                break
            elif lo == "/clear":
                self.memory.clear()
                print("  [Memory cleared — session history preserved]")
            elif lo == "/sources":
                self._show_sources()
            elif lo == "/history":
                self._show_history()
            elif lo == "/report":
                self._cmd_report()
            elif lo == "/backend":
                self._switch_backend()
            elif lo == "/scope":
                self._switch_scope()
            elif lo.startswith("/compare"):
                parts = user.split(maxsplit=1)
                if len(parts) > 1:
                    vlist = [v.strip() for v in parts[1].split(",") if v.strip()]
                else:
                    vlist = discover_vendors()
                if len(vlist) < 2:
                    print("  Need at least 2 vendors. Try: /compare Vendor_A,Vendor_B")
                else:
                    self._set_compare(vlist)
            elif lo.startswith("/vendor"):
                parts = user.split(maxsplit=1)
                self._set_vendor(parts[1] if len(parts) > 1 else "")
            elif lo == "/feedback":
                self.feedback_mode = not self.feedback_mode
                print(f"  Feedback loop: {'ON' if self.feedback_mode else 'OFF'}")
            elif lo == "/mode":
                self.strict_mode = not self.strict_mode
                print(f"  Mode: {'STRICT' if self.strict_mode else 'NORMAL'}")
            elif lo.startswith("/compliance"):
                self._compliance_mode()
            elif lo.startswith("/help"):
                self._show_help()
            elif self._quick_scope_switch(user):
                pass
            else:
                self.ask(user)
                print(build_reference_block(self.last_chunks))
                print(f"\n  [{self.memory.turn_count()} turn(s) | {self.llm.usage()}]")
                if self.feedback_mode:
                    fb, correction = ask_feedback(self.session)
                    if fb == "wrong" and correction:
                        # Inject correction into conversation memory so next turn sees it
                        self.memory.add(
                        "assistant",
                        f"[CORRECTION from reviewer] The previous answer was wrong. "
                        f"Correct assessment: {correction}"
                        )
                print()

    # ── compliance Q&A mode ───────────────────────────────────────────────────
    def _compliance_mode(self):
        print("\n  ── Compliance Q&A Mode ───────────────────────────────────")
        print("  Enter clause no. + requirement. Type 'done' to exit.\n")

        while True:
            try:
                clause_no = input("  Clause No.: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if clause_no.lower() in ("done", "exit", "quit", ""):
                print("  [Exiting compliance mode]")
                break
            try:
                requirement = input("  Requirement: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not requirement:
                continue

            self.ask(f"Does the vendor comply with: {requirement}", clause_no=clause_no)
            print(build_reference_block(self.last_chunks))
            print(f"\n  Clause {clause_no} | {self.llm.usage()}")
            fb, correction = ask_feedback(self.session)
            if fb == "wrong" and correction:
                      self.memory.add(
                             "assistant",
                             f"[CORRECTION from reviewer] The previous answer was wrong. "
                             f"Correct assessment: {correction}"
                      )
            print()

        stats = self.session.accuracy_stats()
        print(f"\n  Session so far: {stats['correct']} correct / "
              f"{stats['wrong']} wrong / {stats['pending']} pending\n")

    # ── show sources ──────────────────────────────────────────────────────────
    def _show_sources(self):
        if not self.last_chunks:
            print("  Ask a question first.")
            return
        print(f"\n  Full retrieved chunks ({len(self.last_chunks)}):")
        print("  " + "─"*56)
        for i, c in enumerate(self.last_chunks, 1):
            vendor  = c.get("vendor", "")
            source  = c.get("source_file", "?")
            page    = c.get("page", "?")
            section = c.get("section", "") or "General"
            score   = c.get("score", 0)
            ctype   = c.get("chunk_type", "text")
            tconf   = c.get("tess_confidence", 0)
            econf   = c.get("easy_confidence", 0)
            ocr_c   = max(tconf, econf)
            v_tag   = f"[{vendor}] " if vendor else ""
            ocr_tag = f"  ⚠ OCR:{ocr_c:.0f}%" if 0 < ocr_c < 70 else ""

            print(f"\n  [{i}] {v_tag}{source} | p.{page} | {section}")
            print(f"       type:{ctype}  score:{score:.4f}{ocr_tag}")
            print(
                "       "
                + textwrap.fill(
                    c["text"][:350],
                    width=56,
                    subsequent_indent="       ",
                )
            )
            if len(c["text"]) > 350:
                print("       …(truncated)")

    # ── show history ──────────────────────────────────────────────────────────
    def _show_history(self):
        turns = self.session.turns
        if not turns:
            print("  No turns yet.")
            return
        print(f"\n  ── Session History ({len(turns)} turns) ───────────────────")
        for t in turns:
            icon   = {"correct":"✓","wrong":"✗","skip":"—","pending":"○"}.get(t["feedback"],"○")
            clause = f"[{t['clause_no']}] " if t.get("clause_no") else ""
            print(f"\n  {icon} Turn {t['turn']} {clause}")
            print(f"     Q: {t['question'][:80]}")
            print(f"     A: {t['answer'][:120]}…")
            if t.get("correction"):
                print(f"     ✏ Correction: {t['correction'][:80]}")
            if t["sources"]:
                src = t["sources"][0]
                print(f"     📄 {src['file']} p.{src['page']} (score:{src['score']:.3f})")
        print()

    # ── generate report ───────────────────────────────────────────────────────
    def _cmd_report(self):
        if not self.session.turns:
            print("  No turns yet — ask some questions first.")
            return
        print_session_report(self.session)
        path = generate_session_report(self.session, vendor=self.active_vendor or "")
        print(f"  JSON report → {path.resolve()}")
        print("  Run compliance_report_generator.py to convert to Excel.\n")
        session_path = self.session.save()
        print(f"  Session  → {session_path.resolve()}\n")

    # ── help ──────────────────────────────────────────────────────────────────
    def _show_help(self):
        print(f"""
  ── Commands ────────────────────────────────────────────────
  /vendor <name>   Restrict retrieval to one vendor's docs.
                   /vendor  (no arg) clears the filter.
  /compare <a,b>   Compare two (or more) vendors side-by-side on every
                   question. No args = compare all detected vendors.
  /scope           Reopen the vendor-scope picker (vendor / compare / all),
                   same menu shown at startup.
  Quick switches   Just type a vendor name (e.g. "Vendor_A"), "compare",
                   "compare all", or "all vendors" on its own — no slash
                   needed for these.
  /backend         Switch between Gemini and Ollama mid-session.
  /compliance      Clause-by-clause compliance Q&A mode.
  /report          Print summary + save compliance_results JSON.
  /history         Show all turns this session with feedback icons.
  /sources         Show raw retrieved chunks from the last answer.
  /feedback        Toggle feedback prompt after answers (default: ON).
  /clear           Wipe conversation memory (session history kept).
  /mode            Toggle strict-citation mode.
  /quit            Save session JSON + exit.
  ────────────────────────────────────────────────────────────
  After each answer:
    [y]  Correct      → marked ✓
    [n]  Wrong        → type correction → saved ✏
    [s]  Skip         → marked —
  ────────────────────────────────────────────────────────────
  Current backend : {self.llm.backend_name}
  Current scope   : {' vs '.join(self.compare_vendors) if self.compare_vendors else (self.active_vendor or 'All')}
        """)

    # ── exit ──────────────────────────────────────────────────────────────────
    def _exit(self):
        print("\n  Saving session…")
        path  = self.session.save()
        stats = self.session.accuracy_stats()
        print(f"  Session saved → {path.resolve()}")
        print(
            f"  Accuracy: {stats['accuracy']}  "
            f"({stats['correct']} correct / {stats['wrong']} wrong / "
            f"{stats['pending']} unrated)"
        )
        print(f"  {self.llm.usage()}")
        print("  Goodbye.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ONGC Tender Chatbot — Gemini + Ollama dual backend"
    )
    parser.add_argument(
        "--backend", choices=["gemini", "ollama"], default=None,
        help="Force a backend (skips the startup menu).",
    )
    parser.add_argument(
        "--vendor", type=str, default=None,
        help="Pre-select a vendor filter (e.g. Vendor_A).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all detected vendors side-by-side on every question.",
    )
    parser.add_argument(
        "--no-feedback", action="store_true",
        help="Disable the per-answer feedback prompt.",
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None,
        help=f"Override Ollama model name (default: {OLLAMA_MODEL}).",
    )
    args = parser.parse_args()

    # Allow model override
    if args.ollama_model:
        OLLAMA_MODEL = args.ollama_model  # type: ignore

    # Select backend (shows menu if --backend not given)
    backend = select_backend(force=args.backend)

    # Build and run chatbot
    bot = TenderChatbot(backend=backend)

    # Decide vendor scope: forced via flag, or interactively if vendors/ has
    # multiple sub-folders and neither --vendor nor --compare was passed.
    detected_vendors = discover_vendors()
    single_vendor, compare_list = select_vendor_mode(
        detected_vendors,
        force_vendor  = args.vendor,
        force_compare = args.compare,
    )
    if compare_list:
        bot._set_compare(compare_list)
    elif single_vendor:
        bot._set_vendor(single_vendor)

    if args.no_feedback:
        bot.feedback_mode = False

    bot.run_cli()
