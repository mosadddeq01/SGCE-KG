







#!############################################# Start Chapter ##################################################
#region:#!   SGCE-KG Generator




#!############################################# Start Chapter ##################################################
#region:#!   Entity Identification



#?######################### Start ##########################
#region:#?   test harness

import traceback
import inspect
import time

def RUN(fn, *args, **kwargs):
    """
    Universal one-line function runner.
    Works for ANY function.
    """
    name = fn.__name__
    print("\n" + "="*80)
    print(f"RUNNING: {name}")
    print("-"*80)

    try:
        sig = inspect.signature(fn)
        print(f"Signature: {name}{sig}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")

        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = time.time() - t0

        print("-"*80)
        print(f"OUTPUT ({dt:.3f}s):")
        print(out)

        return out

    except Exception as e:
        print("-"*80)
        print(f"ERROR in {name}: {e}")
        traceback.print_exc()
        return None

def CHECK(cond, msg="Check failed"):
    if not cond:
        raise AssertionError(msg)


#endregion#? test harness
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   From pdf to JSON file

import re
import json
from pathlib import Path
import pdfplumber

# -------------------------
# Heuristics / regexes
# -------------------------
RE_COPYRIGHT = re.compile(r'copyright|provided by ihs|no reproduction|not for resale', re.I)
RE_BLANK_PAGE = re.compile(r'this page intentionally left blank', re.I)
RE_SECTION_NUMBER = re.compile(r'^\s*(\d+(\.\d+)*)\s+([A-Z][\w \-]{1,})')  # "1.3 Organization and Use"
RE_HEADING_ALLCAPS = re.compile(r'^[A-Z0-9][A-Z0-9 \-\/]{3,}$')  # all caps headings
RE_FIGURE = re.compile(r'^(figure|table)\s*\d+[\-–—]?\s*', re.I)
RE_SHORT_LINE = re.compile(r'^[^\w]{0,4}$')  # useless short lines
RE_TOC_LINE = re.compile(r'^(table of contents|contents)$', re.I)
RE_PAGE_FOOTER = re.compile(r'^\s*[-_]{2,}|^page\s+\d+|\d+/\d+$', re.I)

# Robust TOC detection regex: lines that look like TOC entries (section numbers + dotted leader + page ref)
RE_TOC_ENTRY = re.compile(r'\b\d+(\.\d+)*\b.*\.*\.*\s*\d+[\-–—]?\w*$', re.I)

# Config: what to keep (keywords / section names we prioritize)
KEEP_KEYWORDS = [
    'description', 'affected materials', 'critical factors', 'appearance', 'prevention',
    'inspection', 'monitoring', 'references', 'appendix', 'procedure', 'method', 'failure',
    'damage', 'cause', 'symptom', 'recommend', 'inspection and monitoring'
]

def clean_lines(lines):
    """Filter out obvious noise lines and collapse whitespace."""
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if RE_COPYRIGHT.search(s):
            continue
        if RE_BLANK_PAGE.search(s):
            continue
        if RE_PAGE_FOOTER.search(s) and len(s.split()) < 6:
            continue
        kept.append(s)
    return kept

def is_heading(line):
    if RE_SECTION_NUMBER.match(line):
        return True
    if RE_HEADING_ALLCAPS.match(line) and len(line) < 80:
        return True
    # lines that look like "4.2.9 Thermal Fatigue" or "Thermal Fatigue"
    if len(line.split()) <= 5 and any(k.lower() in line.lower() for k in ['fatigue', 'damage', 'mechanism','introduction','scope','appendix']):
        return True
    return False

def contains_keep_keyword(text):
    tl = text.lower()
    return any(k in tl for k in KEEP_KEYWORDS)

def looks_like_toc_page(lines):
    """
    Heuristic: a page is a TOC page if a significant fraction of its non-empty lines
    look like TOC entries (numbered item + optional dotted leader + page ref).
    """
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return False
    toc_like = sum(1 for ln in nonempty if RE_TOC_ENTRY.search(ln))
    # If more than 30% of lines or at least 4 matches, treat as TOC
    return toc_like >= max(4, int(0.3 * len(nonempty)))

# -------------------------
# Main extraction function
# -------------------------
def extract_relevant_text(pdf_path: str, out_txt: str = "kept_text.txt", out_json: str = "kept_sections.json"):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    sections = []  # each section: {title, start_page, end_page, text, kind}
    current_section = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            # fallback: if page has images or non-extractable text, keep a small marker
            if not raw.strip():
                # check if page contains any images (pdfplumber)
                if page.images:
                    raw = "[IMAGE PAGE] Contains figures/diagrams; extracted metadata only."
                else:
                    raw = ""

            lines = raw.splitlines()
            lines = clean_lines(lines)

            # quick skip if page has little useful text
            if not lines:
                continue

            # Detect whether this page looks like a TOC/index page
            page_is_toc = looks_like_toc_page(lines)
            # If it literally has a "TABLE OF CONTENTS" heading somewhere, force it
            if any(RE_TOC_LINE.search(ln) for ln in lines[:6]):
                page_is_toc = True

            # If this is a TOC page: either start/continue the canonical TOC section
            if page_is_toc:
                toc_text = "\n".join(lines).strip()
                if current_section and current_section.get("title", "").strip().lower() == "table of contents":
                    # append to existing TOC
                    current_section["text"] += "\n" + toc_text
                    current_section["end_page"] = p_idx
                else:
                    # close previous section if any
                    if current_section:
                        current_section["end_page"] = p_idx - 1
                        sections.append(current_section)
                    # start a new canonical TOC section
                    current_section = {
                        "title": "TABLE OF CONTENTS",
                        "start_page": p_idx,
                        "end_page": p_idx,
                        "text": toc_text,
                        "kind": "toc"
                    }
                # we've handled this page, move to next
                continue

            # detect top-of-section headings (first non-noise line)
            found_heading = None
            heading_line_idx = 0
            for i, ln in enumerate(lines[:5]):  # check first few lines for heading
                if is_heading(ln):
                    found_heading = ln.strip()
                    heading_line_idx = i
                    break

            # detect figure/table caption anywhere
            figure_lines = [ln for ln in lines if RE_FIGURE.match(ln)]
            if figure_lines:
                # create a figure entry per caption
                for cap in figure_lines:
                    sec = {
                        "title": cap.strip(),
                        "start_page": p_idx,
                        "end_page": p_idx,
                        "text": cap.strip(),
                        "kind": "figure_caption"
                    }
                    sections.append(sec)

            page_text = "\n".join(lines).strip()

            # If heading detected -> start a new section
            if found_heading:
                # close previous section
                if current_section:
                    current_section["end_page"] = p_idx - 1
                    sections.append(current_section)
                # start a new one using heading as title
                current_section = {
                    "title": found_heading,
                    "start_page": p_idx,
                    "end_page": p_idx,  # will update later
                    "text": "\n".join(lines[heading_line_idx:]).strip(),
                    "kind": "section"
                }
            else:
                # no clear heading: append to current section if it exists
                if current_section:
                    current_section["text"] += "\n" + page_text
                    current_section["end_page"] = p_idx
                else:
                    # no current section: create a provisional block if it contains useful keywords
                    if contains_keep_keyword(page_text) or len(page_text) > 400:
                        blocks = {
                            "title": f"Page {p_idx} block",
                            "start_page": p_idx,
                            "end_page": p_idx,
                            "text": page_text,
                            "kind": "page_block"
                        }
                        sections.append(blocks)
                    else:
                        # skip obvious TOC/short boilerplate
                        if not RE_TOC_LINE.search(page_text):
                            # keep small blocks that include keywords
                            if contains_keep_keyword(page_text):
                                sections.append({
                                    "title": f"Page {p_idx} - small",
                                    "start_page": p_idx,
                                    "end_page": p_idx,
                                    "text": page_text,
                                    "kind": "small_block"
                                })
        # finalize open section
        if current_section:
            sections.append(current_section)

    # Post-filtering: remove tiny sections that are boilerplate
    filtered = []
    for s in sections:
        t = s["text"].strip()
        if len(t) < 80 and s["kind"] not in ("figure_caption", "toc"):
            # keep if contains key keywords
            if not contains_keep_keyword(t):
                continue
        # Drop copyright-only sections
        if RE_COPYRIGHT.search(t) and len(t) < 200:
            continue
        filtered.append(s)

    # Save outputs
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    # plain text aggregation
    with open(out_txt, "w", encoding="utf-8") as ftxt:
        for s in filtered:
            ftxt.write(f"--- {s['title']} (pages {s['start_page']}-{s['end_page']}) [{s['kind']}]\n")
            ftxt.write(s["text"].strip() + "\n\n")

    return {"n_sections": len(filtered), "json": out_json, "txt": out_txt}

# -------------------------
# Quick test runner (single line invocation)
# -------------------------
if __name__ == "__main__":
    res = extract_relevant_text("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/API 571.pdf")
    print("Done. Summary:", res)

RUN(extract_relevant_text, "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/API 571.pdf")


#endregion#?  From pdf to JSON file
#?#########################  End  ##########################



#*######################### Start ##########################
#region:#?   from json text to chunks V0



import json
import re
from pathlib import Path

SUBSECTION_PATTERNS = [
    r"Description of Damage",
    r"Affected Materials",
    r"Critical Factors",
    r"Affected Units or Equipment",
    r"Appearance or Morphology of Damage",
    r"Prevention / Mitigation",
    r"Inspection and Monitoring",
    r"Related Mechanisms",
    r"References"
]

SUBSECTION_REGEX = re.compile(
    "(" + "|".join(SUBSECTION_PATTERNS) + ")",
    re.IGNORECASE
)

def split_by_subsections(text: str):
    """
    Split text while keeping subsection headers.
    """
    parts = SUBSECTION_REGEX.split(text)
    chunks = []
    current_title = None

    for part in parts:
        p = part.strip()
        if not p:
            continue
        if SUBSECTION_REGEX.match(p):
            current_title = p
        else:
            chunks.append((current_title, p))
    return chunks


def chunk_long_text(text, max_chars=1800, overlap_chars=300):
    """
    Fallback chunking for long text blocks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end - overlap_chars
        if start < 0:
            start = 0
    return chunks


def build_chunks_from_pdf_json(
    json_path,
    out_path="chunks.json",
    source_doc="API 571"
):
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    chunks = []
    chunk_id = 0

    for sec in sections:
        if sec["kind"] != "section":
            continue

        title = sec["title"]
        text = sec["text"]

        # skip pure references blocks
        if title.lower().startswith("references"):
            continue

        sub_chunks = split_by_subsections(text)

        if not sub_chunks:
            sub_chunks = [(None, text)]

        for subsection, content in sub_chunks:
            if len(content) < 200:
                continue

            if len(content) > 2000:
                pieces = chunk_long_text(content)
            else:
                pieces = [content]

            for p in pieces:
                chunks.append({
                    "chunk_id": f"Ch_{chunk_id:06d}",
                    "text": p.strip(),
                    "section_title": title,
                    "subsection": subsection,
                    "start_page": sec["start_page"],
                    "end_page": sec["end_page"],
                    "source_doc": source_doc,
                    "kind": "content_chunk"
                })
                chunk_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return {
        "n_chunks": len(chunks),
        "out": out_path
    }

RUN(
    build_chunks_from_pdf_json,
    # "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections.json"
    # "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEditedALL.json"
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEdited_SmallForChunkingTest.json"
)



#endregion#? from json text to chunks V0
#*#########################  End  ##########################



#*######################### Start ##########################
#region:#?   Statistic and information about the input (API 571)

# load this file and compute WORD-based statistics
import json

# Load the JSON file
with open(
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEdited.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

# Number of sections
num_sections = len(data)

# Word counts per section
word_counts = [
    len(section["text"].split())
    for section in data
    if section.get("text", "").strip()
]

# Aggregate stats
total_words = sum(word_counts)
avg_words = total_words / num_sections if num_sections > 0 else 0
min_words = min(word_counts) if word_counts else 0
max_words = max(word_counts) if word_counts else 0

# Print statistics
print(f"Number of sections: {num_sections}")
print(f"Total number of words (all sections): {total_words}")
print(f"Average words per section: {avg_words:.2f}")
print(f"Minimum words in a section: {min_words}")
print(f"Maximum words in a section: {max_words}")














import json
import math
import statistics
from pathlib import Path
import re
from collections import Counter, defaultdict

# small sentence splitter used for many places
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\(\[]|\d)', flags=re.M)

def simple_sent_tokenize(text: str):
    if not text or not text.strip():
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) <= 1 and ("\n" in text):
        # fallback to newline split if no proper punctuation split
        sents = [line.strip() for line in text.splitlines() if line.strip()]
    # last fallback: split on periods (very rough)
    if len(sents) == 0:
        sents = [p.strip() for p in text.split('.') if p.strip()]
    return sents

def _percentile(sorted_list, q):
    """Return q-th percentile (0-100) of a sorted numeric list (pure python)."""
    if not sorted_list:
        return None
    k = (len(sorted_list)-1) * (q/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[int(f)] * (c - k)
    d1 = sorted_list[int(c)] * (k - f)
    return d0 + d1

# default keywords from your earlier code (kept for relevance counts)
KEEP_KEYWORDS = [
    'description', 'affected materials', 'critical factors', 'appearance', 'prevention',
    'inspection', 'monitoring', 'references', 'appendix', 'procedure', 'method', 'failure',
    'damage', 'cause', 'symptom', 'recommend', 'inspection and monitoring'
]

def compute_section_stats(
    sections_json_path: str,
    *,
    save_json: bool = True,
    out_path: str = "data/section_stats_summary.json",
    top_n: int = 10,
    keywords: list = None
):
    """
    Compute rich statistics over a kept_sections JSON file.
    Returns a dict 'summary' and optionally writes to out_path.
    Use RUN(compute_section_stats, path) to execute interactively.
    """
    keywords = keywords if keywords is not None else KEEP_KEYWORDS
    p = Path(sections_json_path)
    assert p.exists(), f"file not found: {p}"

    with open(p, "r", encoding="utf-8") as fh:
        sections = json.load(fh)

    per_section = []
    kinds = Counter()
    title_counts = Counter()
    pages_covered = set()
    sections_with_keywords = 0
    img_only_sections = 0
    empty_sections = 0

    for i, s in enumerate(sections):
        text = s.get("text","") or ""
        title = s.get("title","").strip()
        kind = s.get("kind", "unknown")
        start_page = s.get("start_page")
        end_page = s.get("end_page")
        # tokens = naive split on whitespace
        words = text.split()
        n_words = len(words)
        n_chars = len(text)
        sents = simple_sent_tokenize(text)
        n_sents = len(sents)
        # mark pages
        if isinstance(start_page, int):
            pages_covered.add(start_page)
        if isinstance(end_page, int):
            pages_covered.add(end_page)
        # flags
        if not text.strip():
            empty_sections += 1
        if text.strip().startswith("[IMAGE PAGE]") and n_words <= 6:
            img_only_sections += 1

        # keyword presence
        lower = text.lower()
        has_keyword = any(k.lower() in lower for k in keywords)
        if has_keyword:
            sections_with_keywords += 1

        per_section.append({
            "index": i,
            "title": title,
            "kind": kind,
            "start_page": start_page,
            "end_page": end_page,
            "n_words": n_words,
            "n_sentences": n_sents,
            "n_chars": n_chars,
        })

        kinds[kind] += 1
        title_counts[title] += 1

    # prepare numeric lists
    word_list = sorted([x["n_words"] for x in per_section])
    sent_list = sorted([x["n_sentences"] for x in per_section])
    char_list = sorted([x["n_chars"] for x in per_section])

    def _agg_stats(lst):
        if not lst:
            return {
                "count": 0, "sum": 0, "mean": None, "median": None,
                "min": None, "max": None, "p10": None, "p25": None, "p75": None, "p90": None
            }
        return {
            "count": len(lst),
            "sum": sum(lst),
            "mean": statistics.mean(lst),
            "median": statistics.median(lst),
            "min": lst[0],
            "max": lst[-1],
            "p10": _percentile(lst, 10),
            "p25": _percentile(lst, 25),
            "p75": _percentile(lst, 75),
            "p90": _percentile(lst, 90)
        }

    stats = {
        "file": str(p),
        "n_sections": len(per_section),
        "empty_sections": empty_sections,
        "image_only_sections": img_only_sections,
        "sections_with_keywords": sections_with_keywords,
        "kinds_counts": dict(kinds),
        "pages_covered_count": len(pages_covered),
        "title_duplicate_count": sum(1 for t,c in title_counts.items() if c>1),
        "most_common_titles": title_counts.most_common(10),
        "top_small_sections_by_words": [x for x in per_section if x["n_words"] == 0][:top_n],
        "word_stats": _agg_stats(word_list),
        "sentence_stats": _agg_stats(sent_list),
        "char_stats": _agg_stats(char_list),
        "largest_sections_by_words": sorted(per_section, key=lambda x: x["n_words"], reverse=True)[:top_n],
        "smallest_sections_by_words": sorted(per_section, key=lambda x: x["n_words"])[:top_n],
        "overview_examples": {
            "largest_titles": [ (x["title"], x["n_words"]) for x in sorted(per_section, key=lambda z: z["n_words"], reverse=True)[:5] ],
            "smallest_titles": [ (x["title"], x["n_words"]) for x in sorted(per_section, key=lambda z: z["n_words"])[:5] ],
        }
    }

    # Suggest chunking heuristics
    def recommend_chunking(stats):
        # Prefer sentence-based chunking if sentences are reasonably sized,
        # otherwise prefer word-based chunking. Provide both suggestions.
        w_median = stats["word_stats"]["median"] if stats["word_stats"]["median"] is not None else 0
        w_p75 = stats["word_stats"]["p75"] or 0
        s_median = stats["sentence_stats"]["median"] or 0

        # Basic rules (heuristic):
        # - If median section < 300 words -> keep whole sections (no chunking)
        # - If median section between 300-800 -> chunk into ~250 word windows with 50 word overlap
        # - If median section > 800 -> chunk into ~400 word windows with 100 overlap
        # Also produce sentence-based suggestion: target chunk ~4-8 sentences with 1-2 overlap depending on median
        rec = {}
        if w_median < 300:
            rec["words"] = {
                "mode": "per_section",
                "reason": f"median section words {w_median:.0f} is small => prefer no chunking per section"
            }
        elif w_median < 800:
            rec["words"] = {
                "mode": "sliding_window",
                "chunk_size_words": 250,
                "overlap_words": 50,
                "reason": f"median {w_median:.0f} => moderate size; use 250w chunks with 50w overlap"
            }
        else:
            rec["words"] = {
                "mode": "sliding_window",
                "chunk_size_words": 400,
                "overlap_words": 100,
                "reason": f"median {w_median:.0f} => large sections; use 400w chunks with 100w overlap"
            }

        # sentence-based option
        if s_median == 0:
            rec["sentences"] = {
                "mode": "fallback",
                "chunk_size_sentences": 4,
                "overlap_sentences": 1,
                "reason": "no sentences detected reliably; fallback to 4-sentence chunks"
            }
        elif s_median <= 3:
            rec["sentences"] = {
                "mode": "per_section",
                "reason": f"median sentences per section {s_median:.0f} small => keep per-section"
            }
        elif s_median <= 8:
            rec["sentences"] = {
                "mode": "sliding_window",
                "chunk_size_sentences": 4,
                "overlap_sentences": 1,
                "reason": f"median {s_median:.0f} => 4-sentence chunks with 1 overlap"
            }
        else:
            rec["sentences"] = {
                "mode": "sliding_window",
                "chunk_size_sentences": 6,
                "overlap_sentences": 2,
                "reason": f"median {s_median:.0f} => 6-sentence chunks with 2 overlap"
            }

        # Safety note when many tiny sections or many huge sections exist -> mixed strategy
        if stats["word_stats"]["p90"] and stats["word_stats"]["p90"] > 1200 and stats["word_stats"]["p25"] and stats["word_stats"]["p25"] < 80:
            rec["mixed_strategy"] = (
                "Dataset is skewed (some very large sections & many tiny ones). "
                "Use per-section for small sections and sliding-window for large sections. "
                "Threshold: e.g., sections with >600 words -> sliding-window; else keep per-section."
            )

        return rec

    stats["chunking_recommendation"] = recommend_chunking(stats)

    if save_json:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)

    return stats

# replace with your actual path
RUN(compute_section_stats, "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEdited.json",
    save_json=True, out_path="data/section_stats_summary.json")


#endregion#? Statistic and information about the input (API 571)
#*########################  End  ##########################



#*######################### Start ##########################
#region:#?   Chunking v1


# ---------- Complete revised sentence_chunks_fast (overlap fixed) ----------
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import re

# Use simple heuristic tokens-per-word multiplier (cheap)
TOKENS_PER_WORD = 1.25  # heuristic; change if you want

def estimate_tokens_fast(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))

HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\s+[\w\-\(\) ]{1,80}$')  # crude heading pattern

def sentence_chunks_fast(
    sections_json_path: str,
    out_path: str = "data/chunks_sentence.jsonl",
    max_sentences: int = 6,
    overlap_sentences: int = 2,
    sentence_per_line: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    keep_ref_text: bool = False,
    strip_leading_headings: bool = True,
    force: bool = False,
    debug: bool = False
) -> List[Dict]:
    """
    Fast, robust sentence-based chunker with correct overlap behavior.
    Key fixes:
      - Ensures sliding windows include requested overlap
      - Avoids an early 'break' that could skip final overlapping window
      - Optional: strip leading heading-like sentences from chunk text
    """
    p = Path(sections_json_path)
    assert p.exists(), f"sections file not found: {p}"
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not force:
        with open(outp, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]

    with open(p, "r", encoding="utf-8") as fh:
        sections = json.load(fh)

    if overlap_sentences >= max_sentences:
        raise ValueError("overlap_sentences must be < max_sentences")

    step = max(1, max_sentences - overlap_sentences)

    chunks: List[Dict] = []
    gid = 1

    for sidx, sec in enumerate(tqdm(sections, desc="Chunking sections")):
        title = sec.get("title", f"section_{sidx}")
        text = sec.get("text", "") or ""
        start_page = sec.get("start_page")
        end_page = sec.get("end_page")

        # split into sentences using spaCy (assumes nlp loaded); fallback if needed
        try:
            doc = nlp(text)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            _SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\(\[]|\d)', flags=re.M)
            sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

        if not sents:
            continue

        n_sents = len(sents)
        # start indices for sliding windows
        starts = list(range(0, n_sents, step))
        # ensure we always include a final window that finishes at end (and keeps overlap)
        if starts:
            last_start = starts[-1]
            if last_start + max_sentences < n_sents:
                # add a final start such that last window ends at n_sents
                final_start = max(0, n_sents - max_sentences)
                if final_start != last_start:
                    starts.append(final_start)

        if debug:
            print(f"SECTION {sidx} title='{title}' n_sents={n_sents} starts={starts}")

        chunk_index = 0
        for start in starts:
            end = min(n_sents, start + max_sentences)
            window = sents[start:end]
            if not window:
                continue

            # optionally strip leading heading-like sentence(s) from the window text
            window_for_text = window.copy()
            stripped_heading = None
            if strip_leading_headings and window_for_text:
                first = window_for_text[0]
                # very permissive heading detection: numericals + short uppercase-ish or '4.2...'
                if HEADING_RE.match(first) or len(first) < 120 and first.strip().endswith("Mechanisms") and any(ch.isdigit() for ch in first):
                    stripped_heading = window_for_text.pop(0)

            chunk_text = ("\n" if sentence_per_line else " ").join(window_for_text).strip()
            n_words = len(chunk_text.split()) if chunk_text else 0
            n_tokens_est = estimate_tokens_fast(chunk_text) if chunk_text else 0

            reason = None
            if max_tokens_per_chunk and n_tokens_est > max_tokens_per_chunk:
                # greedy shrink last sentences (cheap)
                while window_for_text and estimate_tokens_fast(("\n" if sentence_per_line else " ").join(window_for_text)) > max_tokens_per_chunk:
                    if len(window_for_text) == 1:
                        words = window_for_text[0].split()
                        max_words_allowed = max(1, int(max_tokens_per_chunk / TOKENS_PER_WORD))
                        window_for_text[0] = " ".join(words[:max_words_allowed])
                        reason = "truncated_long_sentence_by_words"
                        break
                    window_for_text = window_for_text[:-1]
                chunk_text = ("\n" if sentence_per_line else " ").join(window_for_text).strip()
                n_words = len(chunk_text.split()) if chunk_text else 0
                n_tokens_est = estimate_tokens_fast(chunk_text) if chunk_text else 0

            chunk = {
                "id": f"Ch_{gid:06d}",
                "ref_index": sidx,
                "ref_title": title,
                "start_page": start_page,
                "end_page": end_page,
                "chunk_index_in_section": chunk_index,
                "text": chunk_text,
                "sentences": window,                # full sentences in this window (before optional strip)
                "span": [start, end - 1],
                "n_words": n_words,
                "n_tokens_est": n_tokens_est,
                "stripped_heading": stripped_heading,
                "reason": reason
            }
            if keep_ref_text:
                chunk["ref_text"] = text

            chunks.append(chunk)
            gid += 1
            chunk_index += 1

        # done section

    # write ndjson once (fast)
    with open(outp, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks
# ---------- end revised chunker ----------




RUN(sentence_chunks_fast,
    # "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEdited_SmallForChunkingTest.json",
    # "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEdited_MeduimLength.json",
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEditedALL.json",
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    5,   # max_sentences
    2,   # overlap_sentences
    True,# sentence_per_line
    None,# max_tokens_per_chunk -> None for fastest
    False,# keep_ref_text
    True, # strip_leading_headings
    True, # force
    False) # debug


#endregion#? Chunking v1
#*#########################  End  ##########################






#*######################### Start ##########################
#region:#?   Chunking v2



# Revised chunker with min_tokens_per_chunk and controlled expansion
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import re

TOKENS_PER_WORD = 1.25  # heuristic; change if desired

def estimate_tokens_fast(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))

HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\s+[\w\-\(\) ]{1,80}$')

# -----------------------
# Force spaCy sentence splitter
# -----------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Install: python -m spacy download en_core_web_sm") from e
except Exception as e:
    raise RuntimeError("spaCy is required. Install with `pip install spacy` and the model `python -m spacy download en_core_web_sm`.") from e

def split_sentences_spacy(text: str) -> List[str]:
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def sentence_chunks_fast(
    sections_json_path: str,
    out_path: str = "data/chunks_sentence.jsonl",
    max_sentences: int = 6,
    overlap_sentences: int = 2,
    sentence_per_line: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    min_tokens_per_chunk: Optional[int] = None,
    max_sentences_expansion: Optional[int] = None,
    keep_ref_text: bool = False,
    strip_leading_headings: bool = True,
    force: bool = False,
    debug: bool = False
) -> List[Dict]:
    """
    spaCy splitter + token-aware, sentence-preserving chunker with min token expansion.
    - min_tokens_per_chunk: if a produced chunk is smaller than this, add following sentences
      until the min is reached (or section end or expansion limit).
    - max_sentences_expansion: maximum sentences allowed after expansion (defaults to 2*max_sentences).
    """
    p = Path(sections_json_path)
    assert p.exists(), f"sections file not found: {p}"
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not force:
        with open(outp, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]

    with open(p, "r", encoding="utf-8") as fh:
        sections = json.load(fh)

    if overlap_sentences >= max_sentences:
        raise ValueError("overlap_sentences must be < max_sentences")

    if max_sentences_expansion is None:
        max_sentences_expansion = max_sentences * 2

    step = max(1, max_sentences - overlap_sentences)
    chunks: List[Dict] = []
    gid = 1

    for sidx, sec in enumerate(tqdm(sections, desc="Chunking sections")):
        title = sec.get("title", f"section_{sidx}")
        text = sec.get("text", "") or ""
        start_page = sec.get("start_page")
        end_page = sec.get("end_page")

        sents = split_sentences_spacy(text)
        if not sents:
            continue

        n_sents = len(sents)
        starts = list(range(0, n_sents, step))
        if starts:
            last_start = starts[-1]
            if last_start + max_sentences < n_sents:
                final_start = max(0, n_sents - max_sentences)
                if final_start != last_start:
                    starts.append(final_start)

        if debug:
            print(f"SECTION {sidx} title='{title}' n_sents={n_sents} starts={starts}")

        chunk_index = 0
        for start in starts:
            # initial window bounded by max_sentences
            end = min(n_sents, start + max_sentences)
            window = sents[start:end]
            if not window:
                continue

            # optionally strip leading heading-like sentence(s) from the window text
            window_for_text = window.copy()
            stripped_heading = None
            if strip_leading_headings and window_for_text:
                first = window_for_text[0]
                if HEADING_RE.match(first) or (len(first) < 120 and first.strip().endswith("Mechanisms") and any(ch.isdigit() for ch in first)):
                    stripped_heading = window_for_text.pop(0)

            # compute tokens and enforce max_tokens_per_chunk by trimming whole sentences from the end
            reason = None
            chunk_text = ("\n" if sentence_per_line else " ").join(window_for_text).strip()
            n_tokens_est = estimate_tokens_fast(chunk_text) if chunk_text else 0

            if max_tokens_per_chunk and n_tokens_est > max_tokens_per_chunk:
                # trim sentences from end until fit or only one sentence remains
                while len(window_for_text) > 1 and estimate_tokens_fast(("\n" if sentence_per_line else " ").join(window_for_text)) > max_tokens_per_chunk:
                    window_for_text.pop()
                chunk_text = ("\n" if sentence_per_line else " ").join(window_for_text).strip()
                n_tokens_est = estimate_tokens_fast(chunk_text) if chunk_text else 0
                if n_tokens_est > max_tokens_per_chunk and len(window_for_text) == 1:
                    # single long sentence exceeding limit -> keep whole
                    reason = "single_sentence_exceeds_token_limit_kept_whole"
                else:
                    reason = "shrunk_sentences_to_fit_token_limit"

            # Now enforce min_tokens_per_chunk by expanding with following sentences (never split sentences)
            if min_tokens_per_chunk and n_tokens_est < min_tokens_per_chunk:
                # attempt to add successive sentences after the original end (not beyond section)
                add_idx = start + len(window)  # index of next sentence after original window
                # current window length (after possible trimming) is len(window_for_text)
                while n_tokens_est < min_tokens_per_chunk and add_idx < n_sents and len(window_for_text) < max_sentences_expansion:
                    # append next sentence from section
                    next_sent = sents[add_idx]
                    window_for_text.append(next_sent)
                    add_idx += 1
                    chunk_text = ("\n" if sentence_per_line else " ").join(window_for_text).strip()
                    n_tokens_est = estimate_tokens_fast(chunk_text) if chunk_text else 0
                # record reason if expanded
                if n_tokens_est >= min_tokens_per_chunk:
                    reason = (reason + "|expanded_to_meet_min_tokens") if reason else "expanded_to_meet_min_tokens"
                else:
                    # expansion ended by section end or max expansion limit
                    reason = (reason + "|partial_expansion") if reason else "partial_expansion_to_min_tokens"

            n_words = len(chunk_text.split()) if chunk_text else 0

            chunk = {
                "id": f"Ch_{gid:06d}",
                "ref_index": sidx,
                "ref_title": title,
                "start_page": start_page,
                "end_page": end_page,
                "chunk_index_in_section": chunk_index,
                "text": chunk_text,
                "sentences": window,                # original window sentences (before optional strip)
                "span": [start, start + len(window) - 1],
                "n_words": n_words,
                "n_tokens_est": n_tokens_est,
                "stripped_heading": stripped_heading,
                "reason": reason
            }
            if keep_ref_text:
                chunk["ref_text"] = text

            chunks.append(chunk)
            gid += 1
            chunk_index += 1

    # write ndjson
    with open(outp, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks






sentence_chunks_fast(
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEditedALL.json",
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    max_sentences=5,
    overlap_sentences=2,
    sentence_per_line=True,
    max_tokens_per_chunk=350,     # prevent overly long chunks (None to disable)
    min_tokens_per_chunk=250,     # ensure small chunks get expanded
    max_sentences_expansion=10,   # allow expansion up to this many sentences
    keep_ref_text=False,
    strip_leading_headings=True,
    force=True,
    debug=False
)



#endregion#? Chunking v2
#*#########################  End  ##########################




#?######################### Start ##########################
#region:#?   Chunking v3

#!/usr/bin/env python3
"""
Chunking v3 — token-driven, sentence-preserving chunker

- Uses spaCy (en_core_web_sm) for sentence splitting and WILL RAISE if not installed.
- No fixed sentence-count control anymore.
- Two parameters control chunking:
    * max_tokens_per_chunk (int or None) : preferred upper bound for a chunk (won't split sentences)
    * min_tokens_per_chunk (int or None) : ensure small chunks are expanded to reach this minimum
- Behavior (simple and deterministic):
    - Walk sentences in order. Build a chunk by adding sentences until chunk_tokens >= min_tokens_per_chunk.
    - When adding a sentence would push tokens > max_tokens_per_chunk:
        • If chunk currently empty (single sentence > max): accept the single sentence.
        • Else, if removing the last sentence keeps tokens >= min_tokens_per_chunk, pop it and leave it for next chunk.
        • Otherwise keep the sentence (need to meet min requirement).
    - Never split sentences.
    - No overlap windows (keeps provenance simple).
- Output: JSONL file of chunks with fields similar to your previous format.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import re
from tqdm import tqdm

# ---------- simple token estimator (cheap heuristic) ----------
TOKENS_PER_WORD = 1.25
def estimate_tokens_fast(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))

HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\s+[\w\-\(\) ]{1,80}$')

# -----------------------
# require spaCy + model
# -----------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm") from e
except Exception as e:
    raise RuntimeError("spaCy is required. Install with `pip install spacy` and the model `python -m spacy download en_core_web_sm`.") from e

def split_sentences_spacy(text: str) -> List[str]:
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# -----------------------
# main chunker
# -----------------------
def sentence_chunks_token_driven(
    sections_json_path: str,
    out_path: str = "data/chunks_sentence.jsonl",
    max_tokens_per_chunk: Optional[int] = 350,
    min_tokens_per_chunk: Optional[int] = 250,
    sentence_per_line: bool = True,
    keep_ref_text: bool = False,
    strip_leading_headings: bool = True,
    force: bool = False,
    debug: bool = False
) -> List[Dict]:
    """
    Token-driven chunker. See module docstring for behavior.
    """
    src = Path(sections_json_path)
    assert src.exists(), f"sections file not found: {src}"
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not force:
        with open(outp, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]

    with open(src, "r", encoding="utf-8") as fh:
        sections = json.load(fh)

    chunks: List[Dict] = []
    global_gid = 1

    for sidx, sec in enumerate(tqdm(sections, desc="Chunking sections")):
        title = sec.get("title", f"section_{sidx}")
        text = sec.get("text", "") or ""
        start_page = sec.get("start_page")
        end_page = sec.get("end_page")

        sents = split_sentences_spacy(text)
        if not sents:
            continue

        n_sents = len(sents)
        if debug:
            print(f"[section {sidx}] title={title!r} n_sents={n_sents}")

        sent_idx = 0
        chunk_index_in_section = 0

        while sent_idx < n_sents:
            # Start a new chunk
            chunk_sentences: List[str] = []
            chunk_tokens = 0
            chunk_start_idx = sent_idx
            stripped_heading = None

            # If strip_leading_headings: check next sentence and remove it from chunk text if heading-like,
            # but keep it in provenance (we will still include it in 'sentences' for traceability).
            # Implementation: if heading, we will not remove it from chunk_sentences list but will not include in chunk_text.
            # Simpler: we detect heading and treat it as normal sentence but set stripped_heading field if popped later.
            # For simplicity, we will remove a leading heading-like sentence from chunk_text, but keep it as part of 'sentences' for provenance.
            # However, to keep behavior consistent, we'll implement: if the very next sentence looks like a heading, record and skip it from chunk_text,
            # and increment sent_idx by 1 (heading consumed and not part of token/accounting).
            if strip_leading_headings and sent_idx < n_sents:
                maybe = sents[sent_idx]
                if HEADING_RE.match(maybe) or (len(maybe) < 120 and maybe.strip().endswith("Mechanisms") and any(ch.isdigit() for ch in maybe)):
                    stripped_heading = maybe
                    # consume heading from stream (it won't count toward tokens or sentences in chunk text)
                    sent_idx += 1
                    # if we've consumed all sentences, break
                    if sent_idx >= n_sents:
                        # create an empty chunk that only contains heading? Better to emit a small chunk with heading text
                        chunk_sentences = []
                        chunk_text = maybe
                        chunk_tokens = estimate_tokens_fast(chunk_text)
                        # finalize single-heading chunk
                        chunk = {
                            "id": f"Ch_{global_gid:06d}",
                            "ref_index": sidx,
                            "ref_title": title,
                            "start_page": start_page,
                            "end_page": end_page,
                            "chunk_index_in_section": chunk_index_in_section,
                            "text": chunk_text,
                            "sentences": [maybe],
                            "span": [chunk_start_idx, chunk_start_idx],
                            "n_words": len(chunk_text.split()),
                            "n_tokens_est": chunk_tokens,
                            "stripped_heading": stripped_heading,
                            "reason": "heading_only"
                        }
                        if keep_ref_text:
                            chunk["ref_text"] = text
                        chunks.append(chunk)
                        global_gid += 1
                        chunk_index_in_section += 1
                        break  # move to next section

            # Now build chunk by adding sentences until we meet min_tokens_per_chunk (if set),
            # respecting max_tokens_per_chunk when possible.
            while sent_idx < n_sents:
                next_sent = sents[sent_idx]
                next_tok = estimate_tokens_fast(next_sent)

                # If chunk currently empty:
                if not chunk_sentences:
                    # tentatively add the sentence
                    chunk_sentences.append(next_sent)
                    chunk_tokens += next_tok
                    sent_idx += 1
                    # if this single sentence already exceeds max_tokens_per_chunk and min exists, we keep it
                    if max_tokens_per_chunk and chunk_tokens > max_tokens_per_chunk:
                        # single sentence exceeds max - we accept it per policy
                        break
                    # otherwise continue to accumulate until we hit min (if min set)
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        break  # satisfied
                    # if no min is set and we have a sensible size, we can also break to avoid tiny chunks:
                    if not min_tokens_per_chunk and max_tokens_per_chunk and chunk_tokens >= max_tokens_per_chunk:
                        break
                    # else continue loop to add more
                    continue

                # chunk has content already; check if adding next sentence would exceed max
                would_be = chunk_tokens + next_tok
                if max_tokens_per_chunk and would_be > max_tokens_per_chunk:
                    # decide whether to keep next_sent in this chunk or postpone it
                    # compute tokens without last added sentence (i.e., current chunk_tokens)
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        # we already satisfy min; leave next_sent for the next chunk
                        break
                    else:
                        # we do NOT satisfy min yet. Add the sentence even if it exceeds max (to meet min)
                        chunk_sentences.append(next_sent)
                        chunk_tokens = would_be
                        sent_idx += 1
                        break
                else:
                    # safe to add next sentence
                    chunk_sentences.append(next_sent)
                    chunk_tokens = would_be
                    sent_idx += 1
                    # if we have a min and satisfied it, we can stop
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        break
                    # otherwise loop to add more (or until n_sents)
                    continue

            # finalize chunk_text (may exclude stripped_heading from text if present)
            chunk_text = ("\n" if sentence_per_line else " ").join(chunk_sentences).strip()
            n_tokens_est = chunk_tokens
            n_words = len(chunk_text.split()) if chunk_text else 0
            span_end = (chunk_start_idx + len(chunk_sentences) - 1) if chunk_sentences else chunk_start_idx

            # reason field for diagnostics
            reason = None
            if max_tokens_per_chunk and n_tokens_est > max_tokens_per_chunk:
                reason = "exceeded_max_kept_whole_or_to_meet_min"
            elif min_tokens_per_chunk and n_tokens_est < min_tokens_per_chunk:
                reason = "below_min_unable_to_expand"  # e.g., reached section end

            chunk = {
                "id": f"Ch_{global_gid:06d}",
                "ref_index": sidx,
                "ref_title": title,
                "start_page": start_page,
                "end_page": end_page,
                "chunk_index_in_section": chunk_index_in_section,
                "text": chunk_text,
                "sentences": chunk_sentences,
                "span": [chunk_start_idx, span_end],
                "n_words": n_words,
                "n_tokens_est": n_tokens_est,
                "stripped_heading": stripped_heading,
                "reason": reason
            }
            if keep_ref_text:
                chunk["ref_text"] = text

            chunks.append(chunk)
            global_gid += 1
            chunk_index_in_section += 1

        # end while sent_idx

    # write ndjson
    with open(outp, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks

# -----------------------
# example run (use these exact values)
# -----------------------
if __name__ == "__main__":
    sentence_chunks_token_driven(
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/kept_sections_manuallyEditedALL.json",
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
        max_tokens_per_chunk=350,   # preferred upper bound (None to disable)
        min_tokens_per_chunk=250,   # expand small chunks to reach this minimum (None to disable)
        sentence_per_line=True,
        keep_ref_text=False,
        strip_leading_headings=True,
        force=True,
        debug=False
    )


#endregion#? Chunking v3
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Embedding + FAISS Index

# ------------------------------
# Chunk Embedding + FAISS Index (fast, batched)
# ------------------------------
import os
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss

@torch.no_grad()
def embed_and_index_chunks(
    chunks_jsonl_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    output_prefix: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_emb",
    embed_model_large: str = "BAAI/bge-large-en-v1.5",
    embed_model_small: str = "BAAI/bge-small-en-v1.5",
    use_small_model_for_dev: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
    save_index: bool = True,
    force: bool = False
):
    """
    Read NDJSON chunks, produce embeddings, save metadata + embeddings + optional FAISS index.
    Returns (meta_list, embeddings, faiss_index or None)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_path = Path(chunks_jsonl_path)
    assert in_path.exists(), f"Chunks file not found: {in_path}"

    out_meta_path = Path(f"{output_prefix}_meta.jsonl")
    out_vec_path = Path(f"{output_prefix}_vecs.npy")
    out_idx_path = Path(f"{output_prefix}_faiss.index")

    # idempotent shortcut
    if out_meta_path.exists() and out_vec_path.exists() and (out_idx_path.exists() or not save_index) and not force:
        # load
        meta = [json.loads(l) for l in open(out_meta_path, "r", encoding="utf-8")]
        vecs = np.load(out_vec_path)
        index = None
        if save_index and out_idx_path.exists():
            index = faiss.read_index(str(out_idx_path))
        return meta, vecs, index

    # choose model
    model_name = embed_model_small if use_small_model_for_dev else embed_model_large
    print(f"Loading tokenizer+model: {model_name} -> device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # load chunks
    with open(in_path, "r", encoding="utf-8") as fh:
        chunks = [json.loads(l) for l in fh]

    # prepare outputs
    metas = []
    vecs_list = []

    # batching
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        texts = [c.get("text","") for c in batch]
        # tokenize
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        # forward
        out = model(**enc)
        # pooling: mean-pool over valid tokens (attention mask)
        if hasattr(out, "last_hidden_state"):
            token_emb = out.last_hidden_state  # (B, T, D)
        elif hasattr(out, "hidden_states"):
            token_emb = out.hidden_states[-1]
        else:
            raise RuntimeError("Model output has no last_hidden_state or hidden_states")

        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        token_emb = token_emb * mask  # zero out padded tokens
        sum_emb = token_emb.sum(dim=1)  # (B, D)
        denom = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_emb = sum_emb / denom  # (B, D)

        # normalize
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)

        emb_np = mean_emb.cpu().numpy().astype("float32")
        vecs_list.append(emb_np)

        for j, c in enumerate(batch):
            metas.append({
                "id": c.get("id"),
                "ref_index": c.get("ref_index"),
                "ref_title": c.get("ref_title"),
                "chunk_index_in_section": c.get("chunk_index_in_section"),
                "start_page": c.get("start_page"),
                "end_page": c.get("end_page"),
                "n_words": c.get("n_words"),
                # minimal provenance to keep meta small
            })

    all_vecs = np.vstack(vecs_list).astype("float32")

    # save metadata and vectors
    with open(out_meta_path, "w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.save(out_vec_path, all_vecs)

    index = None
    if save_index:
        dim = all_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_vecs)
        faiss.write_index(index, str(out_idx_path))

    return metas, all_vecs, index

# Example one-line RUN (adjust flags as needed)
# Use the small model for development (fast). Set use_small_model_for_dev=False to switch to large.
RUN(embed_and_index_chunks,
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_emb",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    False,   # use_small_model_for_dev
    32,     # batch_size
    None,   # device -> auto
    True,   # save_index
    True)  # force


#endregion#? Embedding + FAISS Index
#?#########################  End  ##########################





#*######################### Start ##########################
#region:#?   Entity Recognition V6 - Broader hint better prmpting


import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG: paths ----------
CHUNKS_JSONL =      "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl"
ENTITIES_OUT =      "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
DEFAULT_DEBUG_DIR = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entity_raw_debug_prompts_outputs"

# ---------- OPENAI client (load key from env or fallback file path) ----------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------- Utility: load chunks ----------
def load_chunks(chunks_jsonl_path: str = CHUNKS_JSONL) -> List[Dict]:
    p = Path(chunks_jsonl_path)
    assert p.exists(), f"chunks file not found: {p}"
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]

# ---------- Save helper (append-safe) ----------
def save_entities(entities: List[Dict], out_path: str = ENTITIES_OUT):
    if not entities:
        print("save_entities: nothing to save.")
        return
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "a", encoding="utf-8") as fh:
        for e in entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved {len(entities)} entities to {out_path}")

# ---------- Small helper to find a chunk by id ----------
def get_chunk_by_id(chunk_id: str, chunks: List[Dict]) -> Dict:
    for c in chunks:
        if c.get("id") == chunk_id:
            return c
    raise ValueError(f"chunk_id {chunk_id} not found")

# ---------- Helper: get previous chunks from same section ----------
def get_previous_chunks(chunk: Dict, chunks: List[Dict], prev_n: int = 1) -> List[Dict]:
    """
    Return up to prev_n previous chunks from the same ref_index (by chunk_index_in_section order).
    Preserves chronological order (oldest -> nearest previous).
    """
    if prev_n <= 0:
        return []
    ref_index = chunk.get("ref_index")
    idx_in_section = chunk.get("chunk_index_in_section", 0)
    same_sec = [c for c in chunks if c.get("ref_index") == ref_index]
    same_sec_sorted = sorted(same_sec, key=lambda x: x.get("chunk_index_in_section", 0))
    prevs = []
    for c in same_sec_sorted:
        if c.get("chunk_index_in_section", 0) < idx_in_section:
            prevs.append(c)
    # take at most prev_n from the end (nearest previous), then return in chronological order
    prevs = prevs[-prev_n:] if prevs else []
    return prevs

# ---------- Prompt builder (chunk + optional prev_chunks) ----------
def build_entity_prompt_with_context(chunk: Dict, prev_chunks: Optional[List[Dict]] = None) -> str:
    """
    Build a prompt that includes the focus chunk and n previous chunk(s) as explicit context.
    Previous chunks are concatenated as plain text (no IDs) to form CONTEXT text.
    The LLM's task: extract entities from the FOCUS chunk only. Previous chunks are provided
    as context for disambiguation. The prompt encourages adding low-confidence new entities if
    uncertain (they will be resolved later).

    Notes:
    - The suggested type hints are *recommendations* only. The model is explicitly allowed
      to propose any other, more specific, or domain-appropriate type strings.
    - For mechanisms/processes, explicitly request the canonical short label (e.g., "graphitization")
      even if the text refers to it indirectly ("this type of ...").
    """
    focus_text = chunk.get("text", "") or ""

    # Suggested (preferred) type hints — not exhaustive and NOT binding.
    suggested_types = [
        "Component",
        "Material",
        "DamageMechanism",
        "FailureEvent",
        "Symptom",
        "Action",
        "FunctionalUnit",
        "OperatingCondition",
        "Environment",
        "InspectionMethod",
        "MitigationAction",
        "Location",
        "ProcessUnit",
        "TemporalQualifier",     # e.g., 'during startup', 'after prolonged exposure'
        "SpatialQualifier",      # e.g., 'inner bore', 'heat-affected zone'
        "CausalHint",            # phrases suggesting causality
        "LogicalMarker",         # e.g., 'if', 'when', 'provided that'
        "UncertaintyQualifier"   # e.g., 'may', 'likely', 'suspected'
    ]

    parts = [
        "GOAL: We are creating a context-enriched knowledge graph (KG) from textual documents.",
        "Your task is to extract entity mentions from the FOCUS chunk.",
        "",
        "PRINCIPLES (read carefully):",
        "- If uncertain about a mention, still include it with a LOWER confidence score; we will resolve duplicates later.",
        "- Be broad-minded: the list of type hints below is a helpful suggestion, but DO NOT be constrained by it.",
        "- If you believe a mention belongs to a different or more specific type, propose that type (string) in `entity_type_hint`.",
        "",
        "TASK SUMMARY:",
        "- Extract entities that appear in the FOCUS chunk only (do NOT invent facts not supported by the chunk).",
        "- You may CONSULT the CONTEXT block (previous chunks concatenated) to disambiguate or resolve pronouns.",
        "- Only list mentions present in the FOCUS chunk; if a piece of CONTEXT helped resolve a pronoun or ambiguity, include a short excerpt in `used_context_excerpt` (optional).",
        "",
        "CRITICAL INSTRUCTION FOR CONCEPTUAL ENTITIES (ENTITY-LEVEL, NOT CLASSES):",
        "- For any entity that refers to a recurring concept (e.g., phenomena, processes, failure modes, behaviors, conditions, states, methods), always extract a SHORT, STABLE, and REUSABLE entity-level label as `entity_name`.",
        "- This label is the best canonical *entity-level* surface name for what is referred to in this text; it is NOT an ontology class, NOT a schema concept, and NOT a general category.",
        "- When the text describes the entity indirectly or descriptively (e.g., 'this type of …', 'degradation due to …', 'loss of … under conditions'), infer the most appropriate short entity-level label even if the exact word does not explicitly appear.",
        "- You may (and often should) separate the short entity label (`entity_name`) from its manifestation or evidence, placing descriptive phrases in `entity_description` or `context_phrase`.",
        "- If you are uncertain about the best entity-level label, still propose the most likely one and lower the `confidence_score` accordingly (e.g., 0.5–0.7); do NOT avoid extraction due to uncertainty.",
        "- IMPORTANT: You are extracting entities grounded in this document, not ontology classes or schema elements; abstraction and class induction will be handled in a later stage.",
        "",
        "CONFIDENCE GUIDELINE (short):",
        "- 0.90 - 1.00 : Certain — explicit, unambiguous mention with clear support in the FOCUS chunk.",
        "- 0.70 - 0.89 : Likely — supported by FOCUS chunk or resolved by CONTEXT.",
        "- 0.40 - 0.69 : Possible — plausible interpretation; partial support.",
        "- 0.00 - 0.39 : Speculative — weakly supported or inferred; include only if potentially useful.",
        "",
        "ENTITY DESCRIPTION:",
        "- For each entity provide a short description (10-25 words) derived from the FOCUS chunk and CONTEXT if needed.",
        "- If a reliable description is not possible, keep it concise and reduce the confidence score.",
        "",
        "SUGGESTED TYPE HINTS (prefer these but you may propose others):",
        f"- {', '.join(suggested_types)}",
        "",
        "ADDITIONAL CONTEXT/QUALIFIERS (you may return these as extra fields or use them in descriptions):",
        "- Condition, Symptom, OperationalConstraint, CausalHint, LogicalMarker, UncertaintyQualifier, TemporalQualifier, SpatialQualifier",
        "",
        "OUTPUT FORMAT INSTRUCTIONS (very important):",
        "- Return ONLY a single JSON array (no extra commentary, no markdown fences).",
        "- Each element must be an object with the following keys:",
        "   * entity_name (string): exact surface form or canonical short label from the FOCUS chunk",
        "   * entity_description (string): short 10-25 word description based on FOCUS (and CONTEXT if used)",
        "   * entity_type_hint (string): suggested type (prefer types above) or a better specific type you propose",
        "   * context_phrase (string): 3-10 word excerpt from the FOCUS chunk showing the mention",
        "   * confidence_score (float): 0.0 - 1.0 as per the guideline above",
        "Optional helpful fields (include only if useful):",
        "   * used_context_excerpt (string): short text from the concatenated CONTEXT that helped disambiguate",
        "   * qualifiers (array of strings): e.g., ['TemporalQualifier: during startup', 'CausalHint: due to H2S']",
        "",
        "IMPORTANT:",
        "- DO NOT list entities that appear only in the CONTEXT block. Only extract mentions present in the FOCUS chunk.",
        "- You are allowed and encouraged to propose more specific `entity_type_hint` values rather than forcing a catch-all.",
        "",
        "=== CONTEXT (previous chunks concatenated) ==="
    ]

    # concatenate previous chunks as plain text blocks, separated by a single blank line
    if prev_chunks:
        ctx_texts = [pc.get("text", "").strip() for pc in prev_chunks if pc.get("text", "").strip()]
        if ctx_texts:
            # join without metadata; plain textual context helps LLM resolve pronouns/co-reference
            parts.append("\n\n".join(ctx_texts))
        else:
            parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")
    else:
        parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")

    parts.append("")
    parts.append("=== FOCUS CHUNK (extract from here) ===")
    # include focus chunk id for provenance but instruction emphasizes extraction from text
    parts.append(f"FOCUS_CHUNK_ID: {chunk.get('id')}")
    parts.append(focus_text)
    parts.append("")
    parts.append("EXAMPLE OUTPUT (three diverse examples — strictly follow JSON shape):")

    # three diverse examples: maintenance, software, clinical
    examples = [
      {
        "entity_name": "graphitization",
        "entity_description": "Formation of graphite in the steel matrix that can reduce tensile strength and increase brittleness.",
        "entity_type_hint": "DamageMechanism",
        "context_phrase": "this type of graphitization",
        "confidence_score": 0.95
      },
      {
        "entity_name": "authentication failure",
        "entity_description": "A software event where user credentials are rejected, often logged with repeated attempts and error 401.",
        "entity_type_hint": "SoftwareEvent",
        "context_phrase": "repeated authentication failure for user 'svc_backup'",
        "confidence_score": 0.85
      },
      {
        "entity_name": "chest pain",
        "entity_description": "Patient-reported acute chest pain radiating to the left arm, onset during exertion; possible angina symptom.",
        "entity_type_hint": "ClinicalSymptom",
        "context_phrase": "patient complained of chest pain radiating to left arm",
        "confidence_score": 0.88
      }
    ]
    parts.append(json.dumps(examples, ensure_ascii=False, indent=2))

    # final join
    return "\n\n".join(parts)





# ---------- OpenAI call (wrapper) ----------
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 2000, temperature: float = 0.0) -> str:
    try:
        print(f"[call_openai] model={model} max_tokens={max_tokens} temperature={temperature}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        print(f"[call_openai] received response (len={len(txt)} chars)")
        return txt
    except Exception as e:
        print("OpenAI call error:", e)
        return ""

# ---------- Debug file writer (single clean implementation) ----------
def write_debug_file(debug_dir: str, chunk: Dict, prev_ctx: List[Dict],
                     prompt: str, llm_output: str,
                     parsed_entities: Optional[List[Dict]] = None,
                     error: Optional[str] = None) -> str:
    """
    Write a JSON file containing:
      - explicit focus_chunk (full chunk dict with id, ref_index, chunk_index_in_section, ref_title, text),
      - explicit context_chunks (list of {id, ref_index, chunk_index_in_section, ref_title, text}),
      - prompt_full (string),
      - llm_output_full (string),
      - parsed_entities (list),
      - error (string or None), metadata.
    Returns the path to the file created.
    """
    debug_dir_path = Path(debug_dir)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    ts = datetime.utcnow().isoformat() + "Z"
    fname = f"{chunk.get('id','unknown')}_{ts.replace(':','-').replace('.','-')}_{run_id[:8]}.json"
    out_path = debug_dir_path / fname

    # prepare context chunks minimal view (id + text) to avoid huge nested objects
    context_min = []
    for pc in (prev_ctx or []):
        context_min.append({
            "id": pc.get("id"),
            "ref_index": pc.get("ref_index"),
            "chunk_index_in_section": pc.get("chunk_index_in_section"),
            "ref_title": pc.get("ref_title"),
            "text": pc.get("text")
        })

    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "chunk_id": chunk.get("id"),
        "focus_chunk": {
            "id": chunk.get("id"),
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text": chunk.get("text")
        },
        "context_chunks": context_min,
        "prompt_full": prompt,
        "llm_output_full": llm_output,
        "parsed_entities": parsed_entities or [],
        "error": error
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return str(out_path)

# ---------- Main: extract_entities_from_chunk (with optional prev context + debug saving) ----------
def extract_entities_from_chunk(
    chunk_id: str,
    chunks_path: str = CHUNKS_JSONL,
    prev_chunks: int = 1,       # how many previous chunks to include as CONTEXT (default 1). Set 0 to disable.
    model: str = "gpt-4o",
    max_tokens: int = 800,
    save_debug: bool = False,   # if True, write full prompt+output+parsed to a debug JSON file
    debug_dir: str = DEFAULT_DEBUG_DIR
) -> List[Dict]:
    """
    Extract entities from the specified focus chunk, optionally including up to `prev_chunks`
    previous chunks from the same section as disambiguating CONTEXT.

    If save_debug=True, a structured JSON file containing the full prompt and full LLM output
    (and the focus/context text) will be written to `debug_dir` for later inspection.
    """
    chunks = load_chunks(chunks_path)
    try:
        chunk = get_chunk_by_id(chunk_id, chunks)
    except ValueError as e:
        print(e)
        return []

    prev_ctx = get_previous_chunks(chunk, chunks, prev_n=prev_chunks)

    prompt = build_entity_prompt_with_context(chunk, prev_ctx)

    # debug: show short prompt preview in console (we keep this to avoid huge console dumps)
    p_shown = prompt if len(prompt) <= 2000 else prompt[:2000] + "\n\n...[TRUNCATED PROMPT]"
    print(f"\n--- ENTITY EXTRACTION PROMPT for {chunk_id} (prev_ctx={len(prev_ctx)}) ---\n{p_shown}\n{'-'*80}")

    raw = call_openai(prompt, model=model, max_tokens=max_tokens)
    if not raw:
        print("Empty LLM response.")
        # optionally save debug with empty llm_output
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, "", [], error="Empty LLM response")
            print(f"Debug file written to: {dbg_path}")
        return []

    txt = raw.strip()
    # unwrap markdown fences if present (be liberal)
    if txt.startswith("```") and txt.endswith("```"):
        txt = txt.strip("`")
        txt = txt.replace("json", "", 1).strip()

    # console preview of LLM output (short)
    preview = txt if len(txt) <= 4000 else txt[:4000] + "\n\n...[TRUNCATED OUTPUT]"
    print(f"[LLM raw output preview]\n{preview}\n{'-'*80}")

    parsed = []
    error_msg = None
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list/array")
        print(f"Parsed JSON array with {len(parsed)} items")
    except Exception as e:
        error_msg = str(e)
        print("Failed to parse JSON from model output:", e)
        print("Model raw output (truncated):", txt[:2000])
        # if debug saving enabled, still save the raw output and error (including focus/context)
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, [], error=error_msg)
            print(f"Debug file written to: {dbg_path}")
        return []

    results = []
    for e in parsed:
        name = e.get("entity_name") or e.get("name") or e.get("label")
        if not name:
            continue
        ent = {
            "id": f"En_{uuid.uuid4().hex[:8]}",
            "flag": "entity_raw",
            "chunk_id": chunk_id,
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text_span": e.get("context_phrase"),
            "entity_name": name,
            "entity_description": e.get("entity_description") or "",
            "entity_type_hint": e.get("entity_type_hint") or e.get("type") or "Other",
            "confidence_score": (float(e.get("confidence_score")) if e.get("confidence_score") is not None else None),
            "used_context_ids": e.get("used_context_ids", []),
            "_raw_llm": e
        }
        results.append(ent)

    print(f"extract_entities_from_chunk: extracted {len(results)} canonical entity records (will be saved).")

    # SAVE results into canonical NDJSON file (append-safe)
    save_entities(results, out_path=ENTITIES_OUT)

    # If debug saving is requested, write the full prompt + output + parsed entities to file (including focus+context)
    if save_debug:
        dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, parsed, error=None)
        print(f"Debug file written to: {dbg_path}")

    return results

# ---------- Driver example ----------
# chunk_ids = ["Ch_000120"]  # modify as needed
# chunk_ids = [f"Ch_{i:06d}" for i in range(0, 224)]

chunk_ids = [ 

"Ch_000001", "Ch_000002", "Ch_000003", "Ch_000004", "Ch_000005", "Ch_000006", "Ch_000007", "Ch_000008", "Ch_000009", "Ch_000010",
"Ch_000011", "Ch_000012", "Ch_000013", "Ch_000014", "Ch_000015", "Ch_000016", "Ch_000017", "Ch_000119", "Ch_000120", "Ch_000121", "Ch_000122",
"Ch_000138", "Ch_000139", "Ch_000140", "Ch_000141", "Ch_000142", "Ch_000143", 
]


def run_entity_extraction_on_chunks(chunk_ids, prev_chunks: int = 1, save_debug: bool = False, debug_dir: str = DEFAULT_DEBUG_DIR):
    all_results = []
    for cid in chunk_ids:
        res = extract_entities_from_chunk(cid, CHUNKS_JSONL, prev_chunks=prev_chunks, model="gpt-4o", max_tokens=1800, save_debug=save_debug, debug_dir=debug_dir)
        if res:
            all_results.extend(res)
    return all_results

# Example run:
if __name__ == "__main__":
    # set save_debug=True to persist full prompt+llm output (and focus/context text) to files in DEFAULT_DEBUG_DIR
    run_entity_extraction_on_chunks(chunk_ids, prev_chunks=5, save_debug=False)


#endregion#? Entity Recognition V6 - Broader hint better prmpting
#*#########################  End  ##########################


























#?######################### Start ##########################
#region:#?   Entity Recognition v7 - Intrinsic properties added


import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG: paths ----------
CHUNKS_JSONL =      "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl"
ENTITIES_OUT =      "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
DEFAULT_DEBUG_DIR = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entity_raw_debug_prompts_outputs"

# ---------- OPENAI client (load key from env or fallback file path) ----------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------- Utility: load chunks ----------
def load_chunks(chunks_jsonl_path: str = CHUNKS_JSONL) -> List[Dict]:
    p = Path(chunks_jsonl_path)
    assert p.exists(), f"chunks file not found: {p}"
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]

# ---------- Save helper (append-safe) ----------
def save_entities(entities: List[Dict], out_path: str = ENTITIES_OUT):
    if not entities:
        print("save_entities: nothing to save.")
        return
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "a", encoding="utf-8") as fh:
        for e in entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved {len(entities)} entities to {out_path}")

# ---------- Small helper to find a chunk by id ----------
def get_chunk_by_id(chunk_id: str, chunks: List[Dict]) -> Dict:
    for c in chunks:
        if c.get("id") == chunk_id:
            return c
    raise ValueError(f"chunk_id {chunk_id} not found")

# ---------- Helper: get previous chunks from same section ----------
def get_previous_chunks(chunk: Dict, chunks: List[Dict], prev_n: int = 1) -> List[Dict]:
    """
    Return up to prev_n previous chunks from the same ref_index (by chunk_index_in_section order).
    Preserves chronological order (oldest -> nearest previous).
    """
    if prev_n <= 0:
        return []
    ref_index = chunk.get("ref_index")
    idx_in_section = chunk.get("chunk_index_in_section", 0)
    same_sec = [c for c in chunks if c.get("ref_index") == ref_index]
    same_sec_sorted = sorted(same_sec, key=lambda x: x.get("chunk_index_in_section", 0))
    prevs = []
    for c in same_sec_sorted:
        if c.get("chunk_index_in_section", 0) < idx_in_section:
            prevs.append(c)
    # take at most prev_n from the end (nearest previous), then return in chronological order
    prevs = prevs[-prev_n:] if prevs else []
    return prevs

# ---------- Prompt builder (chunk + optional prev_chunks) ----------
def build_entity_prompt_with_context(chunk: Dict, prev_chunks: Optional[List[Dict]] = None) -> str:
    """
    Build a prompt that includes the focus chunk and n previous chunk(s) as explicit context.
    Previous chunks are concatenated as plain text (no IDs) to form CONTEXT text.
    The LLM's task: extract entities from the FOCUS chunk only. Previous chunks are provided
    as context for disambiguation. The prompt encourages adding low-confidence new entities if
    uncertain (they will be resolved later).

    Notes:
    - The suggested type hints are *recommendations* only. The model is explicitly allowed
      to propose any other, more specific, or domain-appropriate type strings.
    - This version enforces: do NOT extract relation-level qualifiers here (postpone to Rel Rec).
    - The only allowed "properties" now are truly intrinsic node properties — very rare.
    """
    focus_text = chunk.get("text", "") or ""

    # Suggested (preferred) type hints — not exhaustive and NOT binding.
    suggested_types = [
        "Component",
        "Material",
        "DamageMechanism",
        "FailureEvent",
        "Symptom",
        "Action",
        "FunctionalUnit",
        "OperatingCondition",
        "Environment",
        "InspectionMethod",
        "MitigationAction",
        "Location",
        "ProcessUnit",
        "TemporalQualifier",     # e.g., 'during startup', 'after prolonged exposure'
        "SpatialQualifier",      # e.g., 'inner bore', 'heat-affected zone'
        "CausalHint",            # phrases suggesting causality
        "LogicalMarker",         # e.g., 'if', 'when', 'provided that'
        "UncertaintyQualifier"   # e.g., 'may', 'likely', 'suspected'
    ]

    parts = [
        "GOAL: We are creating a context-enriched knowledge graph (KG) from textual documents.",
        "YOUR TASK (THIS STEP ONLY): Extract entity mentions from the FOCUS chunk ONLY. Relation-level qualifiers, conditions, and other contextual information will be extracted later in the RELATION EXTRACTION step (Rel Rec).",
        "",
        "PRINCIPLES (read carefully):",
        "- Extract broadly: prefer recall (extract candidate mentions). Do NOT be conservative. When in doubt, include the candidate mention. Later stages will cluster, canonicalize, and resolve.",
        "- Ground every output in the FOCUS chunk. You may CONSULT CONTEXT (previous chunks concatenated) only for disambiguation/pronoun resolution.",
        "- DO NOT output relation-level qualifiers, situational context, or evidential/epistemic markers in this step. The ONLY exception is truly intrinsic node properties (see 'INTRINSIC NODE PROPERTIES' below).",
        "- The suggested type hints below are guidance — you may propose more specific domain-appropriate types.",
        "",
        "CORE INSTRUCTION FOR CONCEPTUAL ENTITIES (ENTITY-LEVEL, NOT CLASSES):",
        "- For recurring concepts (phenomena, processes, failure modes, behaviors, conditions, states, methods), extract a SHORT, STABLE, REUSABLE entity-level label as `entity_name`.",
        "- `entity_name` is a canonical mention-level surface form (normalized for this mention). It is NOT an ontology or Schema class label. If you think a class is relevant, place it in `entity_type_hint`.",
        "- If the text describes the concept indirectly (e.g., 'this type of …', 'loss of … under conditions'), infer the best short label (e.g., 'graphitization') and put evidence in `entity_description` and `resolution_context`.",
        "- If unsure of the label, still propose it and lower the `confidence_score` (e.g., 0.5–0.7). We prefer 'extract first, judge later'.",
        "",
        "INTRINSIC NODE PROPERTIES (VERY RARE — only include when unavoidable):",
        "- You may include `node_properties` ONLY when the property is identity-defining for the entity (removing it would change what the entity fundamentally is).",
        "- Allowed intrinsic examples (MANDATORY when present in FOCUS): taxonomic/subtype identity (e.g., material_grade='304'), canonical document identifiers, or stable numeric attributes that define identity (e.g., chemical formula).",
        "- Forbidden here (postpone to Rel Rec): temporal conditions, spatial qualifiers, operational constraints, uncertainty/modality, causal hints, evidence types. These are relation/assertion-level and must NOT be returned in this step.",
        "- Expectation: node_properties should occur really rare, as most of the properties are defined in relation to other entities, therefore we postpone them to Rel Rec.",
        "",
        "CONFIDENCE GUIDELINES:",
        "- 0.90 - 1.00 : Certain — explicit mention in FOCUS chunk, clear support.",
        "- 0.70 - 0.89 : Likely — supported by FOCUS or resolved by CONTEXT.",
        "- 0.40 - 0.69 : Possible — plausible inference; partial support.",
        "- 0.00 - 0.39 : Speculative — weakly supported; include only if likely useful.",
        "",
        "SUGGESTED TYPE HINTS (prefer these but you may propose others):",
        f"- {', '.join(suggested_types)}",
        "",
        "OUTPUT FORMAT INSTRUCTIONS (REVISED — REQUIRED):",
        "- Return ONLY a single JSON array (no extra commentary, no markdown fences).",
        "- Each element must be an object with the following keys (exact names):",
        "   * entity_name (string) — short canonical surface label for the mention (mention-level, NOT class).",
        "   * entity_description (string) — 10–25 word description derived from the FOCUS chunk (and CONTEXT if needed).",
        "   * entity_type_hint (string) — suggested type (from list or a better string).",
        "   * context_phrase (string) — short (3–10 word) excerpt from the FOCUS chunk that PROVES the mention provenance (required when possible).",
        "   * resolution_context (string) — minimal 20–120 word excerpt that best explains WHY this mention maps to `entity_name`. Prefer the sentence containing the mention and at most one neighbor sentence; if CONTEXT was required, include up to one supporting sentence from CONTEXT. This is used for clustering/resolution — make it disambiguating (co-mentions, verbs, numerics).",
        "   * confidence_score (float) — 0.0–1.0.",
        "",
        " - OPTIONAL (include ONLY if an intrinsic property is present and unavoidable):",
        "   * node_properties (array of objects) — each: { 'prop_name': str, 'prop_value': str|num, 'justification': str (one-sentence from FOCUS), 'confidence': float }",
        "",
        "IMPORTANT:",
        "- DO NOT list entities that appear only in CONTEXT. Only extract mentions present in the FOCUS chunk.",
        "- DO NOT output relation qualifiers, situational context, causal hints, or uncertainty markers here. Postpone them to Rel Rec.",
        "- Do not output ontology-level class names as `entity_name`. If relevant, place such information in `entity_type_hint` and keep `entity_name` a mention-level label.",
        "- For conceptual entities that are described indirectly, prefer a short canonical mention and keep the descriptive evidence in `entity_description` and `resolution_context`.",
        "",
        "EMBEDDING WEIGHT NOTE (for clustering later):",
        "WEIGHTS = {\"name\": 0.45, \"desc\": 0.25, \"resolution_context\": 0.25, \"type\": 0.05}",
        "Build resolution_context precisely — it is the second-most important signal after name.",
        "",
        "EXAMPLES (follow JSON shape exactly):"
    ]

    # include previous context if provided
    if prev_chunks:
        ctx_texts = [pc.get("text", "").strip() for pc in prev_chunks if pc.get("text", "").strip()]
        if ctx_texts:
            parts.append("=== CONTEXT (previous chunks concatenated for disambiguation) ===\n")
            parts.append("\n\n".join(ctx_texts))
        else:
            parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")
    else:
        parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")

    parts.append("")
    parts.append("=== FOCUS CHUNK (extract from here) ===")
    parts.append(f"FOCUS_CHUNK_ID: {chunk.get('id')}")
    parts.append(focus_text)
    parts.append("")
    parts.append("EXAMPLE OUTPUT (three diverse examples — strictly follow JSON shape):")

    examples = [
      {
        "entity_name": "graphitization",
        "entity_description": "Formation of graphite in steel that reduces ductility and increases brittleness near weld regions.",
        "entity_type_hint": "DamageMechanism",
        "context_phrase": "this type of graphitization",
        "resolution_context": "this type of graphitization observed in low-alloy steels near welds, indicating localized graphite formation associated with embrittlement and loss of toughness.",
        "confidence_score": 0.85
      },
      {
        "entity_name": "authentication failure",
        "entity_description": "A software event where credentials are rejected repeatedly, often producing error 401 in logs.",
        "entity_type_hint": "SoftwareEvent",
        "context_phrase": "repeated authentication failure",
        "resolution_context": "repeated authentication failure for user 'svc_backup' recorded in the log with multiple 401 entries, indicating failed credential validation.",
        "confidence_score": 0.88
      },
      {
        "entity_name": "austenitic stainless steel",
        "entity_description": "A stainless steel family with austenitic crystal structure, commonly designated by grades like 304 or 316.",
        "entity_type_hint": "Material",
        "context_phrase": "austenitic stainless steel (304)",
        "resolution_context": "austenitic stainless steel (304) explicitly mentioned in FOCUS, parenthetical grade '304' identifies the material grade.",
        "confidence_score": 0.95,
        "node_properties": [
          {"prop_name":"material_grade","prop_value":"304","justification":"explicit parenthetical in FOCUS chunk"}
        ]
      }
    ]
    parts.append(json.dumps(examples, ensure_ascii=False, indent=2))

    # final join
    return "\n\n".join(parts)

# ---------- OpenAI call (wrapper) ----------
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 2000, temperature: float = 0.0) -> str:
    try:
        print(f"[call_openai] model={model} max_tokens={max_tokens} temperature={temperature}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        print(f"[call_openai] received response (len={len(txt)} chars)")
        return txt
    except Exception as e:
        print("OpenAI call error:", e)
        return ""

# ---------- Debug file writer (single clean implementation) ----------
def write_debug_file(debug_dir: str, chunk: Dict, prev_ctx: List[Dict],
                     prompt: str, llm_output: str,
                     parsed_entities: Optional[List[Dict]] = None,
                     error: Optional[str] = None) -> str:
    """
    Write a JSON file containing:
      - explicit focus_chunk (full chunk dict with id, ref_index, chunk_index_in_section, ref_title, text),
      - explicit context_chunks (list of {id, ref_index, chunk_index_in_section, ref_title, text}),
      - prompt_full (string),
      - llm_output_full (string),
      - parsed_entities (list),
      - error (string or None), metadata.
    Returns the path to the file created.
    """
    debug_dir_path = Path(debug_dir)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    ts = datetime.utcnow().isoformat() + "Z"
    fname = f"{chunk.get('id','unknown')}_{ts.replace(':','-').replace('.','-')}_{run_id[:8]}.json"
    out_path = debug_dir_path / fname

    # prepare context chunks minimal view (id + text) to avoid huge nested objects
    context_min = []
    for pc in (prev_ctx or []):
        context_min.append({
            "id": pc.get("id"),
            "ref_index": pc.get("ref_index"),
            "chunk_index_in_section": pc.get("chunk_index_in_section"),
            "ref_title": pc.get("ref_title"),
            "text": pc.get("text")
        })

    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "chunk_id": chunk.get("id"),
        "focus_chunk": {
            "id": chunk.get("id"),
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text": chunk.get("text")
        },
        "context_chunks": context_min,
        "prompt_full": prompt,
        "llm_output_full": llm_output,
        "parsed_entities": parsed_entities or [],
        "error": error
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return str(out_path)

# ---------- Main: extract_entities_from_chunk (with optional prev context + debug saving) ----------
def extract_entities_from_chunk(
    chunk_id: str,
    chunks_path: str = CHUNKS_JSONL,
    prev_chunks: int = 1,       # how many previous chunks to include as CONTEXT (default 1). Set 0 to disable.
    model: str = "gpt-4o",
    max_tokens: int = 800,
    save_debug: bool = False,   # if True, write full prompt+output+parsed to a debug JSON file
    debug_dir: str = DEFAULT_DEBUG_DIR
) -> List[Dict]:
    """
    Extract entities from the specified focus chunk, optionally including up to `prev_chunks`
    previous chunks from the same section as disambiguating CONTEXT.

    If save_debug=True, a structured JSON file containing the full prompt and full LLM output
    (and the focus/context text) will be written to `debug_dir` for later inspection.
    """
    chunks = load_chunks(chunks_path)
    try:
        chunk = get_chunk_by_id(chunk_id, chunks)
    except ValueError as e:
        print(e)
        return []

    prev_ctx = get_previous_chunks(chunk, chunks, prev_n=prev_chunks)

    prompt = build_entity_prompt_with_context(chunk, prev_ctx)

    # debug: show short prompt preview in console (we keep this to avoid huge console dumps)
    p_shown = prompt if len(prompt) <= 2000 else prompt[:2000] + "\n\n...[TRUNCATED PROMPT]"
    print(f"\n--- ENTITY EXTRACTION PROMPT for {chunk_id} (prev_ctx={len(prev_ctx)}) ---\n{p_shown}\n{'-'*80}")

    raw = call_openai(prompt, model=model, max_tokens=max_tokens)
    if not raw:
        print("Empty LLM response.")
        # optionally save debug with empty llm_output
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, "", [], error="Empty LLM response")
            print(f"Debug file written to: {dbg_path}")
        return []

    txt = raw.strip()
    # unwrap markdown fences if present (be liberal)
    if txt.startswith("```") and txt.endswith("```"):
        txt = txt.strip("`")
        txt = txt.replace("json", "", 1).strip()

    # console preview of LLM output (short)
    preview = txt if len(txt) <= 4000 else txt[:4000] + "\n\n...[TRUNCATED OUTPUT]"
    print(f"[LLM raw output preview]\n{preview}\n{'-'*80}")

    parsed = []
    error_msg = None
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list/array")
        print(f"Parsed JSON array with {len(parsed)} items")
    except Exception as e:
        error_msg = str(e)
        print("Failed to parse JSON from model output:", e)
        print("Model raw output (truncated):", txt[:2000])
        # if debug saving enabled, still save the raw output and error (including focus/context)
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, [], error=error_msg)
            print(f"Debug file written to: {dbg_path}")
        return []

    results = []
    for e in parsed:
        # robust field extraction with fallbacks
        name = e.get("entity_name") or e.get("name") or e.get("label")
        if not name:
            continue

        # required fields mapping
        entity_description = e.get("entity_description") or e.get("description") or ""
        entity_type_hint = e.get("entity_type_hint") or e.get("type") or "Other"
        context_phrase = e.get("context_phrase") or ""
        resolution_context = e.get("resolution_context") or e.get("used_context_excerpt") or ""
        confidence_raw = e.get("confidence_score") if e.get("confidence_score") is not None else e.get("confidence")
        try:
            confidence_score = float(confidence_raw) if confidence_raw is not None else None
        except Exception:
            confidence_score = None

        node_props_raw = e.get("node_properties") or []
        # normalize node_properties if present
        node_properties = []
        if isinstance(node_props_raw, list):
            for np in node_props_raw:
                if isinstance(np, dict):
                    node_properties.append({
                        "prop_name": np.get("prop_name") or np.get("name"),
                        "prop_value": np.get("prop_value") or np.get("value"),
                        "justification": np.get("justification", ""),
                        "confidence": float(np.get("confidence")) if np.get("confidence") is not None else None
                    })

        ent = {
            "id": f"En_{uuid.uuid4().hex[:8]}",
            "flag": "entity_raw",
            "chunk_id": chunk_id,
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text_span": context_phrase,
            "entity_name": name,
            "entity_description": entity_description,
            "entity_type_hint": entity_type_hint,
            "confidence_score": confidence_score,
            "resolution_context": resolution_context,
            "node_properties": node_properties
        }
        results.append(ent)

    print(f"extract_entities_from_chunk: extracted {len(results)} canonical entity records (will be saved).")

    # SAVE results into canonical NDJSON file (append-safe)
    save_entities(results, out_path=ENTITIES_OUT)

    # If debug saving is requested, write the full prompt + output + parsed entities to file (including focus+context)
    if save_debug:
        dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, parsed, error=None)
        print(f"Debug file written to: {dbg_path}")

    return results

# ---------- Driver example ----------
# chunk_ids = ["Ch_000120"]  # modify as needed
# chunk_ids = [f"Ch_{i:06d}" for i in range(0, 224)]

# ---------- Driver: run on all chunks in the chunks file ----------
# chunks = load_chunks(CHUNKS_JSONL)
# chunk_ids = [c["id"] for c in chunks]

chunk_ids = [ 

"Ch_000001", "Ch_000002", "Ch_000003", "Ch_000004", "Ch_000005", "Ch_000006", "Ch_000007", "Ch_000008", "Ch_000009", "Ch_000010",
# "Ch_000011", "Ch_000012", "Ch_000013", "Ch_000014", "Ch_000015", "Ch_000016", "Ch_000017", "Ch_000119", "Ch_000120", "Ch_000121", "Ch_000122",
# "Ch_000138", "Ch_000139", "Ch_000140", "Ch_000141", "Ch_000142", "Ch_000143", 
]


def run_entity_extraction_on_chunks(chunk_ids, prev_chunks: int = 1, save_debug: bool = False, debug_dir: str = DEFAULT_DEBUG_DIR):
    all_results = []
    for cid in chunk_ids:
        res = extract_entities_from_chunk(cid, CHUNKS_JSONL, prev_chunks=prev_chunks, model="gpt-4o", max_tokens=1800, save_debug=save_debug, debug_dir=debug_dir)
        if res:
            all_results.extend(res)
    return all_results

# Example run:
if __name__ == "__main__":
    # set save_debug=True to persist full prompt+llm output (and focus/context text) to files in DEFAULT_DEBUG_DIR
    run_entity_extraction_on_chunks(chunk_ids, prev_chunks=5, save_debug=False)






#endregion#? Entity Recognition v7 - Intrinsic properties added
#?#########################  End  ##########################




#*######################### Start ##########################
#region:#?   Embedding and clustering recognized entities - Forced HDBSCAN

"""
embed_and_cluster_entities_force_hdbscan.py

Forces HDBSCAN clustering (no fixed-K fallback) with:
  HDBSCAN_MIN_CLUSTER_SIZE = 5
  HDBSCAN_MIN_SAMPLES = 1

UMAP reduction is optional (recommended).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# --- required clustering lib ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required for this script. Install with `pip install hdbscan`") from e

# optional but recommended
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # we will normalize vectors so euclidean ~ cosine

# UMAP (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")
        ctxs.append(safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or "")
        types.append(safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or "")
    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs, types = build_field_texts(entities)
    D_ref = None
    # encode each field (if empty, make zeros)
    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    D_ref = emb_name.shape[1] if emb_name is not None else None

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    print("[compute] encoding type field ...")
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    if D_ref is None and emb_type is not None:
        D_ref = emb_type.shape[1]

    if D_ref is None:
        raise ValueError("All fields empty — cannot embed")

    # helper to make arrays shape (N, D_ref)
    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = USE_UMAP):
    print(f"[cluster] forcing HDBSCAN min_cluster_size={min_cluster_size} min_samples={min_samples} metric={metric} use_umap={use_umap}")
    X = embeddings
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
        else:
            print("[cluster] running UMAP reduction ->", UMAP_N_COMPONENTS, "dims")
            reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                                min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
            X = reducer.fit_transform(X)
            print("[cluster] UMAP done, X.shape=", X.shape)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    # summary
    summary = {}
    for idx, lab in enumerate(labels):
        summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

# ------------------ Main entry (CLI + notebook-safe) ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    # force HDBSCAN (ignore args.use_method)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=USE_UMAP if args.use_umap else False)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")



if __name__ == "__main__":
    import sys
    # Notebook-friendly defaults when running inside ipykernel
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        class Args:
            # change these defaults if you want different notebook behavior
            entities_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
            out_jsonl = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl"
            clusters_summary = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json"
            use_umap = True   # toggle UMAP inside notebook easily by editing this
        args = Args()
        print("[main] running in notebook mode with defaults:")
        print(f"  entities_in     = {args.entities_in}")
        print(f"  out_jsonl        = {args.out_jsonl}")
        print(f"  clusters_summary = {args.clusters_summary}")
        print(f"  use_umap         = {args.use_umap}")
        main_cli(args)
    else:
        # Running as a normal script -> parse CLI args
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")
        parser.add_argument("--out_jsonl", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
        parser.add_argument("--clusters_summary", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json")
        parser.add_argument("--use_umap", action="store_true", help="Enable UMAP reduction before clustering (recommended)")
        parsed = parser.parse_args()
        # convert parsed Namespace to simple Args-like object expected by main_cli
        class ArgsFromCLI:
            entities_in = parsed.entities_in
            out_jsonl = parsed.out_jsonl
            clusters_summary = parsed.clusters_summary
            use_umap = bool(parsed.use_umap)
        args = ArgsFromCLI()
        main_cli(args)


#endregion#? Embedding and clustering recognized entities - Forced HDBSCAN
#*#########################  End  ##########################


#*######################### Start ##########################
#region:#?   Embedding and clustering recognized entities - Forced HDBSCAN - Resolution + Properties Added

"""
embed_and_cluster_entities_force_hdbscan.py

Forces HDBSCAN clustering (no fixed-K fallback) with:
  HDBSCAN_MIN_CLUSTER_SIZE = 5
  HDBSCAN_MIN_SAMPLES = 1

UMAP reduction is optional (recommended).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# --- required clustering lib ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required for this script. Install with `pip install hdbscan`") from e

# optional but recommended
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
WEIGHTS = {"name": 0.35, "desc": 0.25, "ctx": 0.25, "type": 0.15}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # we will normalize vectors so euclidean ~ cosine

# UMAP (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8  #you see over-merged clusters → decrease N_NEIGHBORS (10)
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: resolution_context (preferred) plus serialized node_properties if present
      - type: entity_type_hint
    Node properties are appended into ctx so they contribute to the 'ctx' embedding (no extra weight).
    """
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        # name & desc
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred; fall back to older fields
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # serialize node_properties if present and append to resolution context (so it contributes to ctx embedding)
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    pieces.append(f"{pname}:{pval}" if pname and pval else pname or pval)
            if pieces:
                node_props_text = " | ".join(pieces)

        # combine resolution + node_properties (node properties folded into ctx)
        if resolution and node_props_text:
            combined_ctx = resolution + " ; " + node_props_text
        elif node_props_text:
            combined_ctx = node_props_text
        else:
            combined_ctx = resolution or ""

        ctxs.append(combined_ctx)

        # type field
        types.append(safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or "")

    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs, types = build_field_texts(entities)
    D_ref = None
    # encode each field (if empty, make zeros)
    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    D_ref = emb_name.shape[1] if emb_name is not None else None

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    print("[compute] encoding type field ...")
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    if D_ref is None and emb_type is not None:
        D_ref = emb_type.shape[1]

    if D_ref is None:
        raise ValueError("All fields empty — cannot embed")

    # helper to make arrays shape (N, D_ref)
    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = USE_UMAP):
    print(f"[cluster] forcing HDBSCAN min_cluster_size={min_cluster_size} min_samples={min_samples} metric={metric} use_umap={use_umap}")
    X = embeddings
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
        else:
            print("[cluster] running UMAP reduction ->", UMAP_N_COMPONENTS, "dims")
            reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                                min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
            X = reducer.fit_transform(X)
            print("[cluster] UMAP done, X.shape=", X.shape)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    # summary
    summary = {}
    for idx, lab in enumerate(labels):
        summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

# ------------------ Main entry (CLI + notebook-safe) ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    # force HDBSCAN (ignore args.use_method)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=USE_UMAP if args.use_umap else False)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")



if __name__ == "__main__":
    import sys
    # Notebook-friendly defaults when running inside ipykernel
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        class Args:
            # change these defaults if you want different notebook behavior
            entities_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
            out_jsonl = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl"
            clusters_summary = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json"
            use_umap = True   # toggle UMAP inside notebook easily by editing this
        args = Args()
        print("[main] running in notebook mode with defaults:")
        print(f"  entities_in     = {args.entities_in}")
        print(f"  out_jsonl        = {args.out_jsonl}")
        print(f"  clusters_summary = {args.clusters_summary}")
        print(f"  use_umap         = {args.use_umap}")
        main_cli(args)
    else:
        # Running as a normal script -> parse CLI args
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")
        parser.add_argument("--out_jsonl", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
        parser.add_argument("--clusters_summary", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json")
        parser.add_argument("--use_umap", action="store_true", help="Enable UMAP reduction before clustering (recommended)")
        parsed = parser.parse_args()
        # convert parsed Namespace to simple Args-like object expected by main_cli
        class ArgsFromCLI:
            entities_in = parsed.entities_in
            out_jsonl = parsed.out_jsonl
            clusters_summary = parsed.clusters_summary
            use_umap = bool(parsed.use_umap)
        args = ArgsFromCLI()
        main_cli(args)


#endregion#? Embedding and clustering recognized entities - Forced HDBSCAN - Resolution + Properties Added
#*#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Embedding and clustering recognized entities    -  V3- Forced HDBSCAN - Resolution + Properties Added (type folded into ctx)

"""
embed_and_cluster_entities_force_hdbscan.py

Changes from previous: the entity_type_hint is injected into the `ctx` text
so context = [TYPE:<type>] + resolution_context + serialized node_properties (if present).
We removed the standalone "type" embedding field and only embed: name, desc, ctx.
Everything else left intact.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# --- required clustering lib ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required for this script. Install with `pip install hdbscan`") from e

# optional but recommended
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
# Note: type field removed and its signal is folded into ctx text.
# Remaining weights have been rescaled to sum to 1.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35 }
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # we will normalize vectors so euclidean ~ cosine

# UMAP (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8  # you see over-merged clusters → decrease N_NEIGHBORS
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties if present
    Node properties are appended into ctx so they contribute to the 'ctx' embedding (no extra weight).
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        # name & desc
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred; fall back to older fields
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # entity type (folded into ctx as a hint)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""

        # serialize node_properties if present and append to resolution context (so it contributes to ctx embedding)
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    pieces.append(f"{pname}:{pval}" if pname and pval else (pname or pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        # build ctx: type hint first (if present), then resolution, then props
        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)

    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs = build_field_texts(entities)
    D_ref = None
    # encode each field (if empty, make zeros)
    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    D_ref = emb_name.shape[1] if emb_name is not None else None

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    if D_ref is None:
        raise ValueError("All fields empty — cannot embed")

    # helper to make arrays shape (N, D_ref)
    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = USE_UMAP):
    print(f"[cluster] forcing HDBSCAN min_cluster_size={min_cluster_size} min_samples={min_samples} metric={metric} use_umap={use_umap}")
    X = embeddings
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
        else:
            print("[cluster] running UMAP reduction ->", UMAP_N_COMPONENTS, "dims")
            reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                                min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
            X = reducer.fit_transform(X)
            print("[cluster] UMAP done, X.shape=", X.shape)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

# def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
#     outp = Path(out_jsonl)
#     outp.parent.mkdir(parents=True, exist_ok=True)
#     with open(outp, "w", encoding="utf-8") as fh:
#         for e, lab in zip(entities, labels):
#             out = dict(e)
#             out["_cluster_id"] = int(lab)
#             fh.write(json.dumps(out, ensure_ascii=False) + "\n")
#     # summary
#     summary = {}
#     for idx, lab in enumerate(labels):
#         summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
#     with open(clusters_summary_path, "w", encoding="utf-8") as fh:
#         json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
#     print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

def save_entities_with_clusters(entities: List[Dict],
                                labels: np.ndarray,
                                out_jsonl: str,
                                clusters_summary_path: str,
                                include_fields: List[str] = None,
                                max_field_chars: int = 240):
    """
    Writes:
      - out_jsonl : entity JSONL annotated with _cluster_id (same as before)
      - clusters_summary_path : richer summary where each entity object in a cluster is written
        as a compact single-line JSON (one line per entity object).

    include_fields: list of keys to include in each entity object (defaults below)
    """
    if include_fields is None:
        include_fields = ["entity_name", "entity_type_hint", "entity_description", "resolution_context"]

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # write JSONL with cluster ids (same as before)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Build cluster mapping with compact per-entity JSON strings
    clusters = {}
    for idx, lab in enumerate(labels):
        lab_int = int(lab)
        ent = entities[idx] if idx < len(entities) else {}
        # helper to safely retrieve + trim fields
        def _get_trim(key):
            v = ent.get(key) or ""
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            s = str(v)
            if len(s) > max_field_chars:
                return s[:max_field_chars-3] + "..."
            return s

        obj = {}
        # entity_name (fallback to placeholder)
        obj["entity_name"] = _get_trim("entity_name") or f"En_{idx}"
        if "entity_type_hint" in include_fields:
            obj["entity_type_hint"] = _get_trim("entity_type_hint") or _get_trim("entity_type") or ""
        if "entity_description" in include_fields:
            obj["entity_description"] = _get_trim("entity_description") or _get_trim("description") or ""
        if "resolution_context" in include_fields:
            ctx = ent.get("resolution_context") or ent.get("text_span") or ent.get("context_phrase") or ent.get("used_context_excerpt") or ""
            if isinstance(ctx, (list, dict)):
                ctx = json.dumps(ctx, ensure_ascii=False)
            ctx = str(ctx)
            if len(ctx) > max_field_chars:
                ctx = ctx[:max_field_chars-3] + "..."
            obj["resolution_context"] = ctx

        clusters.setdefault(lab_int, []).append(obj)

    # Now write clusters_summary.json but ensure each entity object is on a single line.
    meta = {"n_entities": len(entities), "n_clusters": len(clusters)}
    # We'll stream-write the JSON to control formatting: pretty top-level, but compact objects
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')

        # iterate clusters in sorted order of cluster id for stability
        cluster_items = sorted(clusters.items(), key=lambda x: x[0])
        for ci, (lab_int, objs) in enumerate(cluster_items):
            fh.write(f'    "{lab_int}": [\n')
            for oi, obj in enumerate(objs):
                # compact JSON for the object (single line)
                obj_json = json.dumps(obj, ensure_ascii=False, separators=(",", ": "))
                # indent two levels inside cluster array
                fh.write(f'      {obj_json}')
                # trailing comma except last object
                if oi < len(objs) - 1:
                    fh.write(",\n")
                else:
                    fh.write("\n")
            # close this cluster array
            fh.write("    ]")
            # trailing comma except last cluster
            if ci < len(cluster_items) - 1:
                fh.write(",\n")
            else:
                fh.write("\n")

        # close clusters and top-level
        fh.write("  }\n")
        fh.write("}\n")

    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path} (compact entity objects per line)")


# ------------------ Main entry (CLI + notebook-safe) ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    # force HDBSCAN (ignore args.use_method)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=USE_UMAP if args.use_umap else False)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")



if __name__ == "__main__":
    import sys
    # Notebook-friendly defaults when running inside ipykernel
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        class Args:
            # change these defaults if you want different notebook behavior
            entities_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
            out_jsonl = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl"
            clusters_summary = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json"
            use_umap = True   # toggle UMAP inside notebook easily by editing this
        args = Args()
        print("[main] running in notebook mode with defaults:")
        print(f"  entities_in     = {args.entities_in}")
        print(f"  out_jsonl        = {args.out_jsonl}")
        print(f"  clusters_summary = {args.clusters_summary}")
        print(f"  use_umap         = {args.use_umap}")
        main_cli(args)
    else:
        # Running as a normal script -> parse CLI args
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")
        parser.add_argument("--out_jsonl", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
        parser.add_argument("--clusters_summary", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/clusters_summary.json")
        parser.add_argument("--use_umap", action="store_true", help="Enable UMAP reduction before clustering (recommended)")
        parsed = parser.parse_args()
        # convert parsed Namespace to simple Args-like object expected by main_cli
        class ArgsFromCLI:
            entities_in = parsed.entities_in
            out_jsonl = parsed.out_jsonl
            clusters_summary = parsed.clusters_summary
            use_umap = bool(parsed.use_umap)
        args = ArgsFromCLI()
        main_cli(args)


#endregion#? Embedding and clustering recognized entities    -  V3- Forced HDBSCAN - Resolution + Properties Added (type folded into ctx)
#?#########################  End  ##########################





#*######################### Start ##########################
#region:#?   Diagnostics for entities_clustered


# Diagnostics for entities_clustered (revised to use resolution_context & type hints)
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
assert IN.exists(), f"{IN} not found"

ents = []
with open(IN, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        ents.append(json.loads(line))

n = len(ents)
labels = [int(e.get("_cluster_id", -1)) for e in ents]
labels_arr = np.array(labels)
n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
n_noise = int((labels_arr == -1).sum())
print(f"n_entities: {n}, n_clusters (excl -1): {n_clusters}, noise_count: {n_noise} ({n_noise/n*100:.1f}%)")

counts = Counter(labels)
# basic distribution (ignore -1 noise)
sizes = sorted([sz for lab, sz in counts.items() if lab != -1], reverse=True) or [0]
print("Cluster size stats: min, median, mean, max =", min(sizes), np.median(sizes), np.mean(sizes), max(sizes))

# top clusters
top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:15]
print("\nTop 15 clusters (label, size):")
for lab, sz in top:
    print(" ", lab, sz)

# build mapping label -> member objects (compact)
by_label = defaultdict(list)
for i, e in enumerate(ents):
    lab = labels[i]
    name = e.get("entity_name") or e.get("entity_description") or e.get("text_span") or f"En_{i}"
    # prefer new fields
    type_hint = e.get("entity_type_hint") or e.get("entity_type") or ""
    desc = e.get("entity_description") or e.get("description") or ""
    ctx = e.get("resolution_context") or e.get("text_span") or e.get("context_phrase") or e.get("used_context_excerpt") or ""
    # compact node_properties display if present
    node_props = e.get("node_properties") or []
    if isinstance(node_props, list) and node_props:
        npieces = []
        for np in node_props:
            if isinstance(np, dict):
                pname = np.get("prop_name") or np.get("name") or ""
                pval = np.get("prop_value") or np.get("value") or ""
                if pname and pval:
                    npieces.append(f"{pname}={pval}")
                elif pname:
                    npieces.append(pname)
                elif pval:
                    npieces.append(str(pval))
        node_props_str = "; ".join(npieces)
    else:
        node_props_str = ""

    by_label[lab].append({
        "entity_name": name,
        "entity_type_hint": type_hint,
        "entity_description": desc,
        "resolution_context": ctx,
        "node_properties": node_props_str,
        "chunk_id": e.get("chunk_id")
    })

print("\nExamples for top clusters:")
for lab, sz in top[:6]:
    print(f"\nCluster {lab} size={sz}:")
    for v in by_label[lab][:8]:
        # print a single compact line per entity
        parts = [f'{v["entity_name"]!s}']
        if v["entity_type_hint"]:
            parts.append(f'[{v["entity_type_hint"]}]')
        if v["entity_description"]:
            parts.append(f'- {v["entity_description"][:120]}')
        if v["resolution_context"]:
            parts.append(f'| ctx: {v["resolution_context"][:120]}')
        if v["node_properties"]:
            parts.append(f'| props: {v["node_properties"]}')
        if v["chunk_id"]:
            parts.append(f'| chunk:{v["chunk_id"]}')
        print("  -", " ".join(parts))

# small clusters count (size <= 2)
small_count = sum(1 for lab, sz in counts.items() if lab != -1 and sz <= 2)
print(f"\nClusters with size <= 2: {small_count}")


#endregion#? Diagnostics for entities_clustered
#*#########################  End  ##########################







#*######################### Start ##########################
#region:#?   Entity Resolution - V100 -  local sub-clustering and chunk-text inclusion.


# orchestrator_with_chunk_texts_v100.py
"""
Entity resolution orchestrator with robust local sub-clustering, chunk-text inclusion,
token safety guard, and tqdm progress bars.

Usage:
  - In notebook: run the file / import and call orchestrate()
  - From CLI: python orchestrator_with_chunk_texts_v100.py [--entities_in ...] [--chunks ...] [--use_umap]

Requirements:
  pip install torch transformers sentencepiece tqdm hdbscan umap-learn scikit-learn openai
"""

import os
import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# transformers embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client loader (reuses your pattern)
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Paths & config ----------------
CLUSTERED_IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")   # input (from previous clustering)
CHUNKS_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl")

ENT_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
LOG_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/resolution_log.jsonl")

WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# local HDBSCAN
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1
LOCAL_HDBSCAN_METRIC = "euclidean"

# UMAP options
LOCAL_USE_UMAP = False   # default OFF for robustness; enable via CLI --use_umap
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when cluster size >= this

# LLM / prompt
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 800

# orchestration thresholds (as requested)
MAX_CLUSTER_PROMPT = 15        # coarse cluster size threshold to trigger local sub-clustering
MAX_MEMBERS_PER_PROMPT = 10    # <= 10 entities per LLM call
TRUNC_CHUNK_CHARS = 400
INCLUDE_PREV_CHUNKS = 0

# token safety
PROMPT_TOKEN_LIMIT = 2200  # rough char/4 estimate threshold

# ---------------- Utility functions ----------------
def load_chunks(chunks_jsonl_path: Path) -> List[Dict]:
    assert chunks_jsonl_path.exists(), f"Chunks file not found: {chunks_jsonl_path}"
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- HF embedder ----------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            # fallback dimension guess
            return np.zeros((0, self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 1024))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

def build_field_texts(entities: List[Dict]):
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")
        ctxs.append(safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or "")
        types.append(safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or "")
    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs, types = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx, emb_type):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings; check your entity fields")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined

# ---------------- robust local_subcluster ----------------
def local_subcluster(cluster_entities: List[Dict],
                     entity_id_to_index: Dict[str, int],
                     all_embeddings: np.ndarray,
                     min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
                     use_umap: bool = LOCAL_USE_UMAP,
                     umap_dims: int = UMAP_DIMS):
    from collections import defaultdict
    from sklearn.preprocessing import normalize as _normalize

    idxs = [entity_id_to_index[e["id"]] for e in cluster_entities]
    X = all_embeddings[idxs]
    X = _normalize(X, axis=1)
    n = X.shape[0]

    if n <= 1:
        return {0: list(cluster_entities)} if n==1 else {-1: []}

    min_cluster_size = min(min_cluster_size, max(2, n))
    if min_samples is None:
        min_samples = max(1, int(min_cluster_size * 0.1))
    else:
        min_samples = min(min_samples, max(1, n-1))

    X_sub = X
    if use_umap and UMAP_AVAILABLE and n >= UMAP_MIN_SAMPLES_TO_RUN:
        n_components = min(umap_dims, max(2, n - 4))  # keep k <= n-4 for safety
        try:
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(UMAP_NEIGHBORS, max(2, n-1)),
                                min_dist=UMAP_MIN_DIST,
                                metric='cosine',
                                random_state=42)
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=LOCAL_HDBSCAN_METRIC,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = defaultdict(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups

# ------------------ LLM helpers ------------------
def call_llm_with_prompt(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------- Prompt building ----------------
PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
Given the following small cohesive group of candidate entity mentions, decide which ones to MERGE into a single canonical entity, which to MODIFY, and which to KEEP.

Return ONLY a JSON ARRAY. Each element must be one of:
- MergeEntities: {{ "action":"MergeEntities", "entity_ids":[...], "canonical_name":"...", "canonical_description":"...", "canonical_type":"...", "rationale":"..." }}
- ModifyEntity: {{ "action":"ModifyEntity", "entity_id":"...", "new_name":"...", "new_description":"...", "new_type_hint":"...", "rationale":"..." }}
- KeepEntity: {{ "action":"KeepEntity", "entity_id":"...", "rationale":"..." }}

Rules:
- Use ONLY the provided information (name/desc/type_hint/confidence/text_span/chunk_text).
- Be conservative: if unsure, KEEP rather than MERGE.
- If merging, ensure merged items truly refer to the same concept.
- Provide short rationale for each action (1-2 sentences).

Group members (id | name | type_hint | confidence | desc | text_span | chunk_text [truncated]):
{members_json}

Return JSON array only (no commentary).
"""

def build_member_with_chunk(m: Dict, chunks_index: Dict[str, Dict]) -> Dict:
    chunk_text = ""
    chunk_id = m.get("chunk_id")
    if chunk_id:
        ch = chunks_index.get(chunk_id)
        if ch:
            ct = ch.get("text","")
            if INCLUDE_PREV_CHUNKS and isinstance(ch.get("chunk_index_in_section", None), int):
                pass
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ",1)[0] + "..."
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": m.get("text_span"),
        "chunk_text": chunk_text
    }

# ------------------ apply actions ----------------
def apply_actions(members: List[Dict], actions: List[Dict], entities_by_id: Dict[str, Dict],
                  canonical_store: List[Dict], log_entries: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for act in (actions or []):
        typ = act.get("action")
        if typ == "MergeEntities":
            ids = act.get("entity_ids", [])
            canonical_name = act.get("canonical_name")
            canonical_desc = act.get("canonical_description", "")
            canonical_type = act.get("canonical_type", "")
            rationale = act.get("rationale", "")
            can_id = "Can_" + uuid.uuid4().hex[:8]
            canonical = {
                "canonical_id": can_id,
                "canonical_name": canonical_name,
                "canonical_description": canonical_desc,
                "canonical_type": canonical_type,
                "source": "LLM_resolution_v100",
                "rationale": rationale,
                "timestamp": ts
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append({"time": ts, "action": "merge", "canonical_id": can_id, "merged_ids": ids, "rationale": rationale})
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale","")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "modify", "entity_id": eid, "rationale": rationale})
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale","")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "keep", "entity_id": eid, "rationale": rationale})
        else:
            log_entries.append({"time": ts, "action": "unknown", "payload": act})

# ------------------ Orchestration main ----------------
def orchestrate():
    print("Loading clustered entities from:", CLUSTERED_IN)
    entities = []
    with open(CLUSTERED_IN, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entities.append(json.loads(line))
    n_entities = len(entities)
    print("Loaded entities:", n_entities)

    print("Loading chunks from:", CHUNKS_JSONL)
    chunks = load_chunks(CHUNKS_JSONL) if CHUNKS_JSONL.exists() else []
    chunks_index = {c.get("id"): c for c in chunks}
    print("Loaded chunks:", len(chunks))

    entities_by_id = {e["id"]: e for e in entities}
    entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_embeddings = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined_embeddings.shape)

    by_cluster = defaultdict(list)
    for e in entities:
        by_cluster[e.get("_cluster_id")].append(e)

    canonical_store = []
    log_entries = []

    cluster_ids = sorted([k for k in by_cluster.keys() if k != -1])
    noise_count = len(by_cluster.get(-1, []))
    print("Clusters to resolve (excluding noise):", len(cluster_ids), "noise_count:", noise_count)

    # outer progress bar over clusters
    with tqdm(cluster_ids, desc="Clusters", unit="cluster") as pbar_clusters:
        for cid in pbar_clusters:
            members = by_cluster[cid]
            size = len(members)
            pbar_clusters.set_postfix(cluster=cid, size=size)
            # decide path: direct prompt chunks OR local sub-cluster
            if size <= MAX_CLUSTER_PROMPT:
                # number of prompts for this coarse cluster
                n_prompts = math.ceil(size / MAX_MEMBERS_PER_PROMPT)
                with tqdm(range(n_prompts), desc=f"Cluster {cid} prompts", leave=False, unit="prompt") as pbar_prompts:
                    for i in pbar_prompts:
                        s = i * MAX_MEMBERS_PER_PROMPT
                        chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            # skip LLM call, conservative
                            for m in chunk:
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = call_llm_with_prompt(prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # large cluster -> local sub-cluster
                subgroups = local_subcluster(members, entity_id_to_index, combined_embeddings,
                                            min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            use_umap=LOCAL_USE_UMAP, umap_dims=UMAP_DIMS)
                # iterate subgroups sorted by size desc
                sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                with tqdm(sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub") as pbar_subs:
                    for sublab, submembers in pbar_subs:
                        subsize = len(submembers)
                        pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                        if sublab == -1:
                            for m in submembers:
                                m["resolved_action"] = "kept_noise_local"
                                m["resolution_rationale"] = "Local-subcluster noise preserved"
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"keep_noise_local", "entity_id": m["id"], "cluster": cid})
                            continue
                        # prompts for this subcluster
                        n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                        with tqdm(range(n_prompts), desc=f"Sub {sublab} prompts", leave=False, unit="prompt") as pbar_sub_prompts:
                            for i in pbar_sub_prompts:
                                s = i * MAX_MEMBERS_PER_PROMPT
                                chunk = submembers[s:s+MAX_MEMBERS_PER_PROMPT]
                                payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                est_tokens = max(1, int(len(prompt) / 4))
                                pbar_sub_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                                if est_tokens > PROMPT_TOKEN_LIMIT:
                                    for m in chunk:
                                        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                            "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                            "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                        m["resolved_action"] = "kept_skipped_prompt"
                                        m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                    continue
                                llm_out = call_llm_with_prompt(prompt)
                                actions = extract_json_array(llm_out)
                                if actions is None:
                                    actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                                apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"keep_noise_global", "entity_id": ent["id"]})

    # write outputs
    final_entities = list(entities_by_id.values())
    ENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for e in final_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(CANON_OUT, "w", encoding="utf-8") as fh:
        for c in canonical_store:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(LOG_OUT, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    print("\nResolution finished. Wrote:", ENT_OUT, CANON_OUT, LOG_OUT)

# ------------------ Notebook/CLI entry ----------------
if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        print("[main] Running in notebook mode with defaults.")
        orchestrate()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default=str(CLUSTERED_IN))
        parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
        parser.add_argument("--out_entities", type=str, default=str(ENT_OUT))
        parser.add_argument("--canon_out", type=str, default=str(CANON_OUT))
        parser.add_argument("--log_out", type=str, default=str(LOG_OUT))
        parser.add_argument("--use_umap", action="store_true", help="Enable local UMAP inside sub-clustering (only for large clusters)")
        parser.add_argument("--prompt_token_limit", type=int, default=PROMPT_TOKEN_LIMIT)
        parser.add_argument("--max_members_per_prompt", type=int, default=MAX_MEMBERS_PER_PROMPT)
        args = parser.parse_args()

        CLUSTERED_IN = Path(args.entities_in)
        CHUNKS_JSONL = Path(args.chunks)
        ENT_OUT = Path(args.out_entities)
        CANON_OUT = Path(args.canon_out)
        LOG_OUT = Path(args.log_out)
        PROMPT_TOKEN_LIMIT = args.prompt_token_limit
        MAX_MEMBERS_PER_PROMPT = args.max_members_per_prompt
        if args.use_umap:
            LOCAL_USE_UMAP = True
        orchestrate()






#endregion#? Entity Resolution - V100 - local sub-clustering and chunk-text inclusion.
#*#########################  End  ##########################




#*######################### Start ##########################
#region:#?   Entity Resolution - V100 - aligned with EntityRec v7 and embedding pipeline


# orchestrator_with_chunk_texts_v100.py
"""
Entity resolution orchestrator aligned with Entity Recognition v7 and the updated embedding
pipeline (name/desc/ctx). Performs local sub-clustering, chunk-text inclusion, token safety guard,
and tqdm progress bars.

Usage:
  - In notebook: import and call orchestrate()
  - From CLI: python orchestrator_with_chunk_texts_v100.py [--entities_in ...] [--chunks ...] [--use_umap]

Important:
  - This script validates the input clustered-entities JSONL for required fields and raises
    informative errors if fields are missing.
"""

import os
import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# transformers embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client loader (reuses your pattern)
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Paths & config ----------------
CLUSTERED_IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")   # input (from previous clustering)
CHUNKS_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl")

ENT_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
LOG_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/resolution_log.jsonl")

# NOTE: weights changed to match embed_and_cluster V3 (name, desc, ctx) — type is folded into ctx.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# local HDBSCAN
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1
LOCAL_HDBSCAN_METRIC = "euclidean"

# UMAP options
LOCAL_USE_UMAP = False   # default OFF for robustness; enable via CLI --use_umap
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when cluster size >= this

# LLM / prompt
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 800

# orchestration thresholds (as requested)
MAX_CLUSTER_PROMPT = 15        # coarse cluster size threshold to trigger local sub-clustering
MAX_MEMBERS_PER_PROMPT = 10    # <= 10 entities per LLM call
TRUNC_CHUNK_CHARS = 400
INCLUDE_PREV_CHUNKS = 0

# token safety
PROMPT_TOKEN_LIMIT = 2200  # rough char/4 estimate threshold

# ---------------- Utility functions ----------------
def load_chunks(chunks_jsonl_path: Path) -> List[Dict]:
    assert chunks_jsonl_path.exists(), f"Chunks file not found: {chunks_jsonl_path}"
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- HF embedder ----------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        # return shape (N, D) where D = model hidden size
        if len(texts) == 0:
            # fallback: use model.config.hidden_size if available
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------- field builder aligned with EntityRec v7 ----------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties (if present)
    This matches the embedding pipeline in embed_and_cluster V3.
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred (new schema) — fallback chain below
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # fold type and node_properties into ctx (as text hints)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
                    elif pval:
                        pieces.append(str(pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    """
    Compute combined normalized embeddings from name, desc, ctx only.
    """
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings; check your entity fields")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ---------------- robust local_subcluster ----------------
def local_subcluster(cluster_entities: List[Dict],
                     entity_id_to_index: Dict[str, int],
                     all_embeddings: np.ndarray,
                     min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
                     use_umap: bool = LOCAL_USE_UMAP,
                     umap_dims: int = UMAP_DIMS):
    from collections import defaultdict
    from sklearn.preprocessing import normalize as _normalize

    idxs = [entity_id_to_index[e["id"]] for e in cluster_entities]
    X = all_embeddings[idxs]
    X = _normalize(X, axis=1)
    n = X.shape[0]

    if n <= 1:
        return {0: list(cluster_entities)} if n==1 else {-1: []}

    min_cluster_size = min(min_cluster_size, max(2, n))
    if min_samples is None:
        min_samples = max(1, int(min_cluster_size * 0.1))
    else:
        min_samples = min(min_samples, max(1, n-1))

    X_sub = X
    if use_umap and UMAP_AVAILABLE and n >= UMAP_MIN_SAMPLES_TO_RUN:
        n_components = min(umap_dims, max(2, n - 4))  # keep k <= n-4 for safety
        try:
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(UMAP_NEIGHBORS, max(2, n-1)),
                                min_dist=UMAP_MIN_DIST,
                                metric='cosine',
                                random_state=42)
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=LOCAL_HDBSCAN_METRIC,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = defaultdict(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups

# ------------------ LLM helpers ------------------
def call_llm_with_prompt(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------- Prompt building ----------------
PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
Given the following small cohesive group of candidate entity mentions, decide which ones to MERGE into a single canonical entity, which to MODIFY, and which to KEEP.

Return ONLY a JSON ARRAY. Each element must be one of:
- MergeEntities: {{ "action":"MergeEntities", "entity_ids":[...], "canonical_name":"...", "canonical_description":"...", "canonical_type":"...", "rationale":"..." }}
- ModifyEntity: {{ "action":"ModifyEntity", "entity_id":"...", "new_name":"...", "new_description":"...", "new_type_hint":"...", "rationale":"..." }}
- KeepEntity: {{ "action":"KeepEntity", "entity_id":"...", "rationale":"..." }}

Rules:
- Use ONLY the provided information (name/desc/type_hint/confidence/resolution_context/text_span/chunk_text).
- Be conservative: if unsure, KEEP rather than MERGE.
- If merging, ensure merged items truly refer to the same concept.
- Provide short rationale for each action (1-2 sentences).

Group members (id | name | type_hint | confidence | desc | text_span | chunk_text [truncated]):
{members_json}

Return JSON array only (no commentary).
"""

def build_member_with_chunk(m: Dict, chunks_index: Dict[str, Dict]) -> Dict:
    """
    Build the member record included in the LLM prompt.
    Uses resolution_context (preferred) as part of text_span fallback and includes truncated chunk_text if available.
    """
    chunk_text = ""
    chunk_id = m.get("chunk_id")
    if chunk_id:
        ch = chunks_index.get(chunk_id)
        if ch:
            ct = ch.get("text","")
            # optionally include previous chunks? currently disabled (INCLUDE_PREV_CHUNKS)
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ",1)[0] + "..."
    # text_span should show the precise mention proof; prefer context_phrase then resolution_context
    text_span = m.get("context_phrase") or m.get("resolution_context") or m.get("text_span") or ""
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": text_span,
        "chunk_text": chunk_text
    }

# ------------------ apply actions ----------------
def apply_actions(members: List[Dict], actions: List[Dict], entities_by_id: Dict[str, Dict],
                  canonical_store: List[Dict], log_entries: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for act in (actions or []):
        typ = act.get("action")
        if typ == "MergeEntities":
            ids = act.get("entity_ids", [])
            canonical_name = act.get("canonical_name")
            canonical_desc = act.get("canonical_description", "")
            canonical_type = act.get("canonical_type", "")
            rationale = act.get("rationale", "")
            can_id = "Can_" + uuid.uuid4().hex[:8]
            canonical = {
                "canonical_id": can_id,
                "canonical_name": canonical_name,
                "canonical_description": canonical_desc,
                "canonical_type": canonical_type,
                "source": "LLM_resolution_v100",
                "rationale": rationale,
                "timestamp": ts
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append({"time": ts, "action": "merge", "canonical_id": can_id, "merged_ids": ids, "rationale": rationale})
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale","")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "modify", "entity_id": eid, "rationale": rationale})
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale","")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "keep", "entity_id": eid, "rationale": rationale})
        else:
            log_entries.append({"time": ts, "action": "unknown", "payload": act})

# ------------------ Orchestration main ----------------
def validate_entities_schema(entities: List[Dict]):
    """
    Ensure every entity has the minimum required fields expected by the new pipeline.
    Raise ValueError with a helpful message listing offending entries if not.
    """
    required = ["id", "entity_name"]  # minimal required; others allowed to be empty but must exist as keys ideally
    problems = []
    for i, e in enumerate(entities):
        missing = [k for k in required if k not in e]
        # also validate that at least one context field exists (resolution_context | context_phrase | text_span)
        context_present = any(k in e and e.get(k) not in (None, "") for k in ("resolution_context","context_phrase","text_span","used_context_excerpt"))
        if missing or not context_present:
            problems.append({"index": i, "id": e.get("id"), "missing_keys": missing, "has_context": context_present, "sample": {k: e.get(k) for k in ["entity_name","entity_type_hint","resolution_context","context_phrase","text_span"]}})
        # ensure confidence_score exists (may be None but field should be present to avoid KeyError later)
        if "confidence_score" not in e:
            e["confidence_score"] = None

    if problems:
        msg_lines = ["Entities schema validation failed — some entries are missing required keys or have no context field:"]
        for p in problems[:10]:
            msg_lines.append(f" - idx={p['index']} id={p['id']} missing={p['missing_keys']} has_context={p['has_context']} sample={p['sample']}")
        if len(problems) > 10:
            msg_lines.append(f" - ... and {len(problems)-10} more problematic entries")
        raise ValueError("\n".join(msg_lines))

def orchestrate():
    print("Loading clustered entities from:", CLUSTERED_IN)
    entities = []
    with open(CLUSTERED_IN, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entities.append(json.loads(line))
    n_entities = len(entities)
    print("Loaded entities:", n_entities)

    # validate schema early and loudly
    validate_entities_schema(entities)

    print("Loading chunks from:", CHUNKS_JSONL)
    chunks = load_chunks(CHUNKS_JSONL) if CHUNKS_JSONL.exists() else []
    chunks_index = {c.get("id"): c for c in chunks}
    print("Loaded chunks:", len(chunks))

    # ensure ids are unique
    ids = [e.get("id") for e in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entity ids found in clustered input — ids must be unique.")

    entities_by_id = {e["id"]: e for e in entities}
    entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    # embedder + combined embeddings (name/desc/ctx)
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_embeddings = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined_embeddings.shape)

    # group by coarse cluster
    by_cluster = defaultdict(list)
    for e in entities:
        by_cluster[e.get("_cluster_id")].append(e)

    canonical_store = []
    log_entries = []

    cluster_ids = sorted([k for k in by_cluster.keys() if k != -1])
    noise_count = len(by_cluster.get(-1, []))
    print("Clusters to resolve (excluding noise):", len(cluster_ids), "noise_count:", noise_count)

    # outer progress bar over clusters
    with tqdm(cluster_ids, desc="Clusters", unit="cluster") as pbar_clusters:
        for cid in pbar_clusters:
            members = by_cluster[cid]
            size = len(members)
            pbar_clusters.set_postfix(cluster=cid, size=size)

            # decide path: direct prompt chunks OR local sub-cluster
            if size <= MAX_CLUSTER_PROMPT:
                # number of prompts for this coarse cluster
                n_prompts = math.ceil(size / MAX_MEMBERS_PER_PROMPT)
                with tqdm(range(n_prompts), desc=f"Cluster {cid} prompts", leave=False, unit="prompt") as pbar_prompts:
                    for i in pbar_prompts:
                        s = i * MAX_MEMBERS_PER_PROMPT
                        chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            # skip LLM call, conservative
                            for m in chunk:
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = call_llm_with_prompt(prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # large cluster -> local sub-cluster
                subgroups = local_subcluster(members, entity_id_to_index, combined_embeddings,
                                            min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            use_umap=LOCAL_USE_UMAP, umap_dims=UMAP_DIMS)
                # iterate subgroups sorted by size desc
                sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                with tqdm(sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub") as pbar_subs:
                    for sublab, submembers in pbar_subs:
                        subsize = len(submembers)
                        pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                        if sublab == -1:
                            for m in submembers:
                                m["resolved_action"] = "kept_noise_local"
                                m["resolution_rationale"] = "Local-subcluster noise preserved"
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"keep_noise_local", "entity_id": m["id"], "cluster": cid})
                            continue
                        # prompts for this subcluster
                        n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                        with tqdm(range(n_prompts), desc=f"Sub {sublab} prompts", leave=False, unit="prompt") as pbar_sub_prompts:
                            for i in pbar_sub_prompts:
                                s = i * MAX_MEMBERS_PER_PROMPT
                                chunk = submembers[s:s+MAX_MEMBERS_PER_PROMPT]
                                payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                est_tokens = max(1, int(len(prompt) / 4))
                                pbar_sub_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                                if est_tokens > PROMPT_TOKEN_LIMIT:
                                    for m in chunk:
                                        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                            "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                            "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                        m["resolved_action"] = "kept_skipped_prompt"
                                        m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                    continue
                                llm_out = call_llm_with_prompt(prompt)
                                actions = extract_json_array(llm_out)
                                if actions is None:
                                    actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                                apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"keep_noise_global", "entity_id": ent["id"]})

    # write outputs
    final_entities = list(entities_by_id.values())
    ENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for e in final_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(CANON_OUT, "w", encoding="utf-8") as fh:
        for c in canonical_store:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(LOG_OUT, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    print("\nResolution finished. Wrote:", ENT_OUT, CANON_OUT, LOG_OUT)

# ------------------ Notebook/CLI entry ----------------
if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        print("[main] Running in notebook mode with defaults.")
        orchestrate()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default=str(CLUSTERED_IN))
        parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
        parser.add_argument("--out_entities", type=str, default=str(ENT_OUT))
        parser.add_argument("--canon_out", type=str, default=str(CANON_OUT))
        parser.add_argument("--log_out", type=str, default=str(LOG_OUT))
        parser.add_argument("--use_umap", action="store_true", help="Enable local UMAP inside sub-clustering (only for large clusters)")
        parser.add_argument("--prompt_token_limit", type=int, default=PROMPT_TOKEN_LIMIT)
        parser.add_argument("--max_members_per_prompt", type=int, default=MAX_MEMBERS_PER_PROMPT)
        args = parser.parse_args()

        CLUSTERED_IN = Path(args.entities_in)
        CHUNKS_JSONL = Path(args.chunks)
        ENT_OUT = Path(args.out_entities)
        CANON_OUT = Path(args.canon_out)
        LOG_OUT = Path(args.log_out)
        PROMPT_TOKEN_LIMIT = args.prompt_token_limit
        MAX_MEMBERS_PER_PROMPT = args.max_members_per_prompt
        if args.use_umap:
            LOCAL_USE_UMAP = True
        orchestrate()

#endregion#? Entity Resolution - V100 - aligned with EntityRec v7 and embedding pipeline
#*#########################  End  ##########################








#?######################### Start ##########################
#region:#?   Final Ent Res - (aligned with EntityRec v7 and embedding pipeline With SubCluster json)

# orchestrator_with_chunk_texts_v100.py
"""
Entity resolution orchestrator aligned with Entity Recognition v7 and the updated embedding
pipeline (name/desc/ctx). Performs local sub-clustering, chunk-text inclusion, token safety guard,
and tqdm progress bars.

Changes (Dec 2025):
 - stricter input validation (requires _cluster_id + context fields)
 - saves local-subcluster summaries per coarse cluster for debugging
 - fallback to "no local sub-clustering" when fragmentation is excessive (keeps members together)
 - more robust min_cluster_size handling and explanatory logging
"""
import os
import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# transformers embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client loader (reuses your pattern)
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Paths & config ----------------
CLUSTERED_IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")   # input (from previous clustering)
CHUNKS_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl")

ENT_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
LOG_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/resolution_log.jsonl")

# NOTE: weights changed to match embed_and_cluster V3 (name, desc, ctx) — type is folded into ctx.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# local HDBSCAN
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1
LOCAL_HDBSCAN_METRIC = "euclidean"

# UMAP options
LOCAL_USE_UMAP = False   # default OFF for robustness; enable via CLI --use_umap
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when cluster size >= this

# LLM / prompt
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 800

# orchestration thresholds (as requested)
MAX_CLUSTER_PROMPT = 15        # coarse cluster size threshold to trigger local sub-clustering
MAX_MEMBERS_PER_PROMPT = 10    # <= 10 entities per LLM call
TRUNC_CHUNK_CHARS = 400
INCLUDE_PREV_CHUNKS = 0

# token safety
PROMPT_TOKEN_LIMIT = 2200  # rough char/4 estimate threshold

# fallback fragmentation rule (tunable)
# if local sub-clustering produces more than len(members)/FALLBACK_FRAG_THRESHOLD_FACTOR non-noise subclusters,
# we will FALLBACK to _not_ using local_subcluster and instead chunk the original coarse cluster directly.
FALLBACK_FRAG_THRESHOLD_FACTOR = 2   # e.g., if > len(members)/2 non-noise subclusters -> fallback

# ---------------- Utility functions ----------------
def load_chunks(chunks_jsonl_path: Path) -> List[Dict]:
    assert chunks_jsonl_path.exists(), f"Chunks file not found: {chunks_jsonl_path}"
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- HF embedder ----------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        # return shape (N, D) where D = model hidden size
        if len(texts) == 0:
            # fallback: use model.config.hidden_size if available
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------- field builder aligned with EntityRec v7 ----------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties (if present)
    This matches the embedding pipeline in embed_and_cluster V3.
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred (new schema) — fallback chain below
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # fold type and node_properties into ctx (as text hints)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
                    elif pval:
                        pieces.append(str(pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    """
    Compute combined normalized embeddings from name, desc, ctx only.
    """
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings; check your entity fields")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ---------------- robust local_subcluster ----------------
def local_subcluster(cluster_entities: List[Dict],
                     entity_id_to_index: Dict[str, int],
                     all_embeddings: np.ndarray,
                     min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
                     use_umap: bool = LOCAL_USE_UMAP,
                     umap_dims: int = UMAP_DIMS):
    """
    Returns: dict[label] -> list[entity_dict]
    Notes:
      - min_cluster_size is treated as a true minimum (HDBSCAN may still mark small noise).
      - We protect against trivial inputs and return fallback single cluster if hdbscan fails.
    """
    from collections import defaultdict
    from sklearn.preprocessing import normalize as _normalize

    idxs = [entity_id_to_index[e["id"]] for e in cluster_entities]
    X = all_embeddings[idxs]
    X = _normalize(X, axis=1)
    n = X.shape[0]

    if n <= 1:
        return {0: list(cluster_entities)} if n==1 else {-1: []}

    # ensure min_cluster_size is sensible (it is a minimum)
    min_cluster_size = max(2, int(min_cluster_size))
    min_cluster_size = min(min_cluster_size, max(2, n))  # cannot be larger than cluster size

    if min_samples is None:
        min_samples = max(1, int(min_cluster_size * 0.1))
    else:
        min_samples = max(1, int(min_samples))

    X_sub = X
    if use_umap and UMAP_AVAILABLE and n >= UMAP_MIN_SAMPLES_TO_RUN:
        n_components = min(umap_dims, max(2, n - 4))  # keep k <= n-4 for stability
        try:
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(UMAP_NEIGHBORS, max(2, n-1)),
                                min_dist=UMAP_MIN_DIST,
                                metric='cosine',
                                random_state=42)
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=LOCAL_HDBSCAN_METRIC,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = defaultdict(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups

# ------------------ LLM helpers ------------------
def call_llm_with_prompt(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------- Prompt building ----------------
PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
Given the following small cohesive group of candidate entity mentions, decide which ones to MERGE into a single canonical entity, which to MODIFY, and which to KEEP.

Return ONLY a JSON ARRAY. Each element must be one of:
- MergeEntities: {{ "action":"MergeEntities", "entity_ids":[...], "canonical_name":"...", "canonical_description":"...", "canonical_type":"...", "rationale":"..." }}
- ModifyEntity: {{ "action":"ModifyEntity", "entity_id":"...", "new_name":"...", "new_description":"...", "new_type_hint":"...", "rationale":"..." }}
- KeepEntity: {{ "action":"KeepEntity", "entity_id":"...", "rationale":"..." }}

Rules:
- Use ONLY the provided information (name/desc/type_hint/confidence/resolution_context/text_span/chunk_text).
- Be conservative: if unsure, KEEP rather than MERGE.
- If merging, ensure merged items truly refer to the same concept.
- Provide short rationale for each action (1-2 sentences).

Group members (id | name | type_hint | confidence | desc | text_span | chunk_text [truncated]):
{members_json}

Return JSON array only (no commentary).
"""

def build_member_with_chunk(m: Dict, chunks_index: Dict[str, Dict]) -> Dict:
    """
    Build the member record included in the LLM prompt.
    Uses resolution_context (preferred) as part of text_span fallback and includes truncated chunk_text if available.
    """
    chunk_text = ""
    chunk_id = m.get("chunk_id")
    if chunk_id:
        ch = chunks_index.get(chunk_id)
        if ch:
            ct = ch.get("text","")
            # optionally include previous chunks? currently disabled (INCLUDE_PREV_CHUNKS)
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ",1)[0] + "..."
    # text_span should show the precise mention proof; prefer context_phrase then resolution_context
    text_span = m.get("context_phrase") or m.get("resolution_context") or m.get("text_span") or ""
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": text_span,
        "chunk_text": chunk_text
    }

# ------------------ apply actions ----------------
def apply_actions(members: List[Dict], actions: List[Dict], entities_by_id: Dict[str, Dict],
                  canonical_store: List[Dict], log_entries: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for act in (actions or []):
        typ = act.get("action")
        if typ == "MergeEntities":
            ids = act.get("entity_ids", [])
            canonical_name = act.get("canonical_name")
            canonical_desc = act.get("canonical_description", "")
            canonical_type = act.get("canonical_type", "")
            rationale = act.get("rationale", "")
            can_id = "Can_" + uuid.uuid4().hex[:8]
            canonical = {
                "canonical_id": can_id,
                "canonical_name": canonical_name,
                "canonical_description": canonical_desc,
                "canonical_type": canonical_type,
                "source": "LLM_resolution_v100",
                "rationale": rationale,
                "timestamp": ts
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append({"time": ts, "action": "merge", "canonical_id": can_id, "merged_ids": ids, "rationale": rationale})
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale","")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "modify", "entity_id": eid, "rationale": rationale})
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale","")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "keep", "entity_id": eid, "rationale": rationale})
        else:
            log_entries.append({"time": ts, "action": "unknown", "payload": act})

# ------------------ Orchestration main ----------------
def validate_entities_schema(entities: List[Dict]):
    """
    Ensure every entity has the minimum required fields expected by the new pipeline.
    Raise ValueError with a helpful message listing offending entries if not.
    """
    required = ["id", "entity_name", "_cluster_id"]  # require cluster id for grouping
    problems = []
    for i, e in enumerate(entities):
        missing = [k for k in required if k not in e]
        # validate that at least one context field exists (resolution_context | context_phrase | text_span | used_context_excerpt)
        context_present = any(k in e and e.get(k) not in (None, "") for k in ("resolution_context","context_phrase","text_span","used_context_excerpt"))
        if missing or not context_present:
            problems.append({"index": i, "id": e.get("id"), "missing_keys": missing, "has_context": context_present, "sample": {k: e.get(k) for k in ["entity_name","entity_type_hint","resolution_context","context_phrase","text_span","_cluster_id"]}})
        # ensure confidence_score exists (may be None but field should be present to avoid KeyError later)
        if "confidence_score" not in e:
            e["confidence_score"] = None

    if problems:
        msg_lines = ["Entities schema validation failed — some entries are missing required keys or have no context field:"]
        for p in problems[:20]:
            msg_lines.append(f" - idx={p['index']} id={p['id']} missing={p['missing_keys']} has_context={p['has_context']} sample={p['sample']}")
        if len(problems) > 20:
            msg_lines.append(f" - ... and {len(problems)-20} more problematic entries")
        raise ValueError("\n".join(msg_lines))

def write_local_subcluster_summary(out_dir: Path, cid: int, subgroups: Dict[int, List[Dict]]):
    """
    Save a compact summary JSON of local subclusters for debugging.
    Format:
      { "cluster_id": cid, "n_entities": N, "n_subclusters": K, "clusters": {"0": [ {id,name,type_hint,desc}, ...], "-1": [...] } }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cluster_id": cid, "n_entities": sum(len(v) for v in subgroups.values()), "n_subclusters": len(subgroups)}
    clusters = {}
    for lab, members in sorted(subgroups.items(), key=lambda x: x[0]):
        clusters[str(lab)] = [{"id": m.get("id"), "entity_name": m.get("entity_name"), "entity_type_hint": m.get("entity_type_hint"), "entity_description": (m.get("entity_description") or "")[:200]} for m in members]
    summary["clusters"] = clusters
    path = out_dir / f"cluster_{cid}_subclusters.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return path

def orchestrate():
    print("Loading clustered entities from:", CLUSTERED_IN)
    entities = []
    with open(CLUSTERED_IN, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entities.append(json.loads(line))
    n_entities = len(entities)
    print("Loaded entities:", n_entities)

    # validate schema early and loudly
    validate_entities_schema(entities)

    print("Loading chunks from:", CHUNKS_JSONL)
    chunks = load_chunks(CHUNKS_JSONL) if CHUNKS_JSONL.exists() else []
    chunks_index = {c.get("id"): c for c in chunks}
    print("Loaded chunks:", len(chunks))

    # ensure ids are unique
    ids = [e.get("id") for e in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entity ids found in clustered input — ids must be unique.")

    entities_by_id = {e["id"]: e for e in entities}
    entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    # embedder + combined embeddings (name/desc/ctx)
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_embeddings = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined_embeddings.shape)

    # group by coarse cluster
    by_cluster = defaultdict(list)
    for e in entities:
        by_cluster[e.get("_cluster_id")].append(e)

    canonical_store = []
    log_entries = []

    cluster_ids = sorted([k for k in by_cluster.keys() if k != -1])
    noise_count = len(by_cluster.get(-1, []))
    print("Clusters to resolve (excluding noise):", len(cluster_ids), "noise_count:", noise_count)

    local_subclusters_dir = ENT_OUT.parent / "local_subclusters"
    local_subclusters_dir.mkdir(parents=True, exist_ok=True)

    # outer progress bar over clusters
    with tqdm(cluster_ids, desc="Clusters", unit="cluster") as pbar_clusters:
        for cid in pbar_clusters:
            members = by_cluster[cid]
            size = len(members)
            pbar_clusters.set_postfix(cluster=cid, size=size)

            # decide path: direct prompt chunks OR local sub-cluster
            if size <= MAX_CLUSTER_PROMPT:
                # number of prompts for this coarse cluster
                n_prompts = math.ceil(size / MAX_MEMBERS_PER_PROMPT)
                with tqdm(range(n_prompts), desc=f"Cluster {cid} prompts", leave=False, unit="prompt") as pbar_prompts:
                    for i in pbar_prompts:
                        s = i * MAX_MEMBERS_PER_PROMPT
                        chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            # skip LLM call, conservative
                            for m in chunk:
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = call_llm_with_prompt(prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # large cluster -> local sub-cluster
                subgroups = local_subcluster(members, entity_id_to_index, combined_embeddings,
                                            min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            use_umap=LOCAL_USE_UMAP, umap_dims=UMAP_DIMS)

                # ALWAYS persist subcluster summary for inspection (even if we fallback)
                subcluster_summary_path = write_local_subcluster_summary(local_subclusters_dir, cid, subgroups)
                print(f"[info] wrote local subcluster summary: {subcluster_summary_path}")

                # compute fragmentation statistic: number of non-noise subclusters
                nonnoise_clusters = [lab for lab in subgroups.keys() if lab != -1]
                num_nonnoise = len(nonnoise_clusters)
                # fallback condition: too fragmented -> prefer sequential chunking of original coarse cluster
                fallback_threshold = max(1, int(len(members) / FALLBACK_FRAG_THRESHOLD_FACTOR))
                fallback = num_nonnoise > fallback_threshold

                if fallback:
                    # FALLBACK: do not use subgroups; process original members sequentially (keeps original local ordering)
                    print(f"[fallback] cluster {cid} had {num_nonnoise} non-noise subclusters for {len(members)} members; falling back to sequential chunking (preserves grouping).")
                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "action":"fallback_no_subcluster", "cluster": cid, "n_members": len(members), "n_nonnoise_subclusters": num_nonnoise})
                    # chunk original members iteratively
                    n_prompts = math.ceil(len(members) / MAX_MEMBERS_PER_PROMPT)
                    with tqdm(range(n_prompts), desc=f"Cluster {cid} fallback prompts", leave=False, unit="prompt") as pbar_prompts:
                        for i in pbar_prompts:
                            s = i * MAX_MEMBERS_PER_PROMPT
                            chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                            payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                            members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                            prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                            est_tokens = max(1, int(len(prompt) / 4))
                            pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                            if est_tokens > PROMPT_TOKEN_LIMIT:
                                for m in chunk:
                                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                        "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                    m["resolved_action"] = "kept_skipped_prompt"
                                    m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                continue
                            llm_out = call_llm_with_prompt(prompt)
                            actions = extract_json_array(llm_out)
                            if actions is None:
                                actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                            apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
                else:
                    # proceed with subgroups as planned
                    sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                    with tqdm(sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub") as pbar_subs:
                        for sublab, submembers in pbar_subs:
                            subsize = len(submembers)
                            pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                            if sublab == -1:
                                for m in submembers:
                                    m["resolved_action"] = "kept_noise_local"
                                    m["resolution_rationale"] = "Local-subcluster noise preserved"
                                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "action":"keep_noise_local", "entity_id": m["id"], "cluster": cid})
                                continue
                            # prompts for this subcluster
                            n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                            with tqdm(range(n_prompts), desc=f"Sub {sublab} prompts", leave=False, unit="prompt") as pbar_sub_prompts:
                                for i in pbar_sub_prompts:
                                    s = i * MAX_MEMBERS_PER_PROMPT
                                    chunk = submembers[s:s+MAX_MEMBERS_PER_PROMPT]
                                    payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                    members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                    prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                    est_tokens = max(1, int(len(prompt) / 4))
                                    pbar_sub_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                                    if est_tokens > PROMPT_TOKEN_LIMIT:
                                        for m in chunk:
                                            log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                                "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                            m["resolved_action"] = "kept_skipped_prompt"
                                            m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                        continue
                                    llm_out = call_llm_with_prompt(prompt)
                                    actions = extract_json_array(llm_out)
                                    if actions is None:
                                        actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                                    apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"keep_noise_global", "entity_id": ent["id"]})

    # write outputs
    final_entities = list(entities_by_id.values())
    ENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for e in final_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(CANON_OUT, "w", encoding="utf-8") as fh:
        for c in canonical_store:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(LOG_OUT, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    print("\nResolution finished. Wrote:", ENT_OUT, CANON_OUT, LOG_OUT)
    print(f"[info] local subcluster summaries (if any) are under: {local_subclusters_dir}")





# ------------------ Notebook/CLI entry ----------------
if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        print("[main] Running in notebook mode with defaults.")
        orchestrate()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default=str(CLUSTERED_IN))
        parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
        parser.add_argument("--out_entities", type=str, default=str(ENT_OUT))
        parser.add_argument("--canon_out", type=str, default=str(CANON_OUT))
        parser.add_argument("--log_out", type=str, default=str(LOG_OUT))
        parser.add_argument("--use_umap", action="store_true", help="Enable local UMAP inside sub-clustering (only for large clusters)")
        parser.add_argument("--prompt_token_limit", type=int, default=PROMPT_TOKEN_LIMIT)
        parser.add_argument("--max_members_per_prompt", type=int, default=MAX_MEMBERS_PER_PROMPT)
        parser.add_argument("--fallback_frag_factor", type=int, default=FALLBACK_FRAG_THRESHOLD_FACTOR,
                            help="Factor controlling fragmentation fallback (lower -> more likely fallback)")
        args = parser.parse_args()

        CLUSTERED_IN = Path(args.entities_in)
        CHUNKS_JSONL = Path(args.chunks)
        ENT_OUT = Path(args.out_entities)
        CANON_OUT = Path(args.canon_out)
        LOG_OUT = Path(args.log_out)
        PROMPT_TOKEN_LIMIT = args.prompt_token_limit
        MAX_MEMBERS_PER_PROMPT = args.max_members_per_prompt
        FALLBACK_FRAG_THRESHOLD_FACTOR = max(1, int(args.fallback_frag_factor))
        if args.use_umap:
            LOCAL_USE_UMAP = True
        orchestrate()


#endregion#? Final Ent Res - (aligned with EntityRec v7 and embedding pipeline With SubCluster json)
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   Analyze_entity_resolution


#!/usr/bin/env python3
"""
analyze_entity_resolution.py

Creates entResAnalysis/ with:
 - merged_groups.json     : mapping canonical_id -> list of member entities (full objects)
 - merged_groups.csv      : flat CSV: canonical_id, member_id, member_name, desc, type, confidence, _cluster_id, resolved_action
 - canonical_summary.csv  : canonical_id, canonical_name, canonical_type, n_members, example_members
 - actions_summary.json   : counts per resolved_action
 - type_distribution.csv  : counts per entity_type_hint (for merged vs kept)
 - charts: merges_hist.png, actions_pie.png

Usage:
  python analyze_entity_resolution.py
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import pandas as pd
import matplotlib.pyplot as plt

# --------- Config / paths ----------
ENT_RES_FILE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_FILE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------- Helpers ----------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                data.append(json.loads(ln))
            except Exception as e:
                print("skip line (json error):", e)
    return data

# --------- Load data ----------
print("Loading files...")
entities = load_jsonl(ENT_RES_FILE) if ENT_RES_FILE.exists() else []
canons = load_jsonl(CANON_FILE) if CANON_FILE.exists() else []

print(f"Loaded {len(entities)} entities, {len(canons)} canonical records")

# index canonical metadata if present
canon_meta = {c.get("canonical_id"): c for c in canons}

# --------- Build merged groups ----------
merged = defaultdict(list)       # canonical_id -> list of entity dicts
unmerged = []                    # entities without canonical_id
for e in entities:
    cid = e.get("canonical_id")
    if cid:
        merged[cid].append(e)
    else:
        unmerged.append(e)

# Save merged_groups.json
with open(OUT_DIR / "merged_groups.json", "w", encoding="utf-8") as fh:
    json.dump({k: v for k,v in merged.items()}, fh, ensure_ascii=False, indent=2)

# Save merged_groups.csv (flat)
csv_fields = ["canonical_id", "canonical_name", "member_id", "member_name", "member_desc",
              "member_type", "confidence_score", "_cluster_id", "resolved_action", "resolution_rationale"]
with open(OUT_DIR / "merged_groups.csv", "w", newline='', encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=csv_fields)
    writer.writeheader()
    for cid, members in merged.items():
        canon_name = canon_meta.get(cid, {}).get("canonical_name", "")
        for m in members:
            writer.writerow({
                "canonical_id": cid,
                "canonical_name": canon_name,
                "member_id": m.get("id"),
                "member_name": m.get("entity_name"),
                "member_desc": m.get("entity_description"),
                "member_type": m.get("entity_type_hint"),
                "confidence_score": m.get("confidence_score"),
                "_cluster_id": m.get("_cluster_id"),
                "resolved_action": m.get("resolved_action"),
                "resolution_rationale": m.get("resolution_rationale","")
            })

# Save canonical_summary.csv
canon_rows = []
for cid, members in merged.items():
    row = {
        "canonical_id": cid,
        "canonical_name": canon_meta.get(cid, {}).get("canonical_name", ""),
        "canonical_type": canon_meta.get(cid, {}).get("canonical_type", ""),
        "n_members": len(members),
        "example_members": " | ".join([m.get("entity_name","") for m in members[:5]])
    }
    canon_rows.append(row)
canon_df = pd.DataFrame(canon_rows).sort_values("n_members", ascending=False)
canon_df.to_csv(OUT_DIR / "canonical_summary.csv", index=False)

# Save actions summary
action_counts = Counter([e.get("resolved_action","<none>") for e in entities])
with open(OUT_DIR / "actions_summary.json", "w", encoding="utf-8") as fh:
    json.dump(action_counts, fh, ensure_ascii=False, indent=2)

# Save type distribution (merged vs unmerged)
def type_counter(list_of_entities):
    c = Counter()
    for e in list_of_entities:
        t = e.get("entity_type_hint") or "<unknown>"
        c[t] += 1
    return c

merged_types = type_counter([m for members in merged.values() for m in members])
unmerged_types = type_counter(unmerged)

# write as CSV table
all_types = sorted(set(list(merged_types.keys()) + list(unmerged_types.keys())))
with open(OUT_DIR / "type_distribution.csv", "w", newline='', encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["type", "merged_count", "unmerged_count"])
    for t in all_types:
        w.writerow([t, merged_types.get(t,0), unmerged_types.get(t,0)])

# --------- Quick stats & charts ----------
# 1) Histogram of canonical cluster sizes
sizes = [len(members) for members in merged.values()]
if len(sizes) == 0:
    print("No merged canonical groups found. Exiting chart generation.")
else:
    plt.figure(figsize=(6,4))
    plt.hist(sizes, bins=range(1, max(sizes)+2), edgecolor='black')
    plt.title("Distribution of canonical cluster sizes (# members)")
    plt.xlabel("members per canonical entity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "merges_hist.png", dpi=150)
    plt.close()

# 2) Pie chart of resolved_action counts (top actions)
actions_df = pd.DataFrame(action_counts.items(), columns=["action","count"]).sort_values("count", ascending=False)
plt.figure(figsize=(6,6))
top_actions = actions_df.head(6)
plt.pie(top_actions["count"], labels=top_actions["action"], autopct="%1.1f%%", startangle=140)
plt.title("Top resolved_action distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "actions_pie.png", dpi=150)
plt.close()

# 3) Top canonical groups CSV (top 50)
canon_df.head(50).to_csv(OUT_DIR / "top50_canonical.csv", index=False)

# 4) A simple mapping file for quick inspection: canonical_id -> [member names]
simple_map = {cid: [m.get("entity_name") for m in members] for cid,members in merged.items()}
with open(OUT_DIR / "canonical_to_members_sample.json", "w", encoding="utf-8") as fh:
    json.dump(simple_map, fh, ensure_ascii=False, indent=2)

# 5) Save unmerged entities (for manual review)
with open(OUT_DIR / "unmerged_entities.jsonl", "w", encoding="utf-8") as fh:
    for e in unmerged:
        fh.write(json.dumps(e, ensure_ascii=False) + "\n")

# ---------- Print short summary ----------
print("Analysis saved to:", OUT_DIR)
print("Counts:")
print(" - total entities:", len(entities))
print(" - canonical groups (merged):", len(merged))
print(" - unmerged entities:", len(unmerged))
print("Top 10 canonical groups (id, size):")
for cid, members in canon_df[["canonical_id","n_members"]].head(10).itertuples(index=False, name=None):
    print(" ", cid, members)

# optional: show top 10 actions
print("Top actions:")
for a,cnt in action_counts.most_common(10):
    print(" ", a, cnt)

print("\nFiles produced (entResAnalysis/):")
for p in sorted(OUT_DIR.iterdir()):
    print(" ", p.name)


#endregion#? Analyze_entity_resolution
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Iterative Ent Res with_per_run_outputs




# ============================================================
# Iterative Embedding + Clustering + Resolution Driver (Robust)
# ============================================================

from pathlib import Path
import json
import shutil
import time
from typing import List, Dict

# ---------------- Iteration control ----------------
DEFAULT_MAX_ITERS = 3
MIN_MERGES_TO_CONTINUE = 1

# ---------------- Paths ----------------
ENT_RAW_SEED = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")

CLUSTERED_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
CANONICAL_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
RESOLVED_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")

ITER_DIR = (ENT_RAW_SEED.parent).parent / "iterative_runs"
ITER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl(path: Path, objs: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in objs:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")

def normalize_entity_schema(e: Dict) -> Dict:
    """
    Force a consistent schema/order for ALL entities.
    """
    return {
        "id": e.get("id"),
        "entity_name": e.get("entity_name"),
        "entity_description": e.get("entity_description"),
        "entity_type_hint": e.get("entity_type_hint"),
        "confidence_score": e.get("confidence_score"),
        "resolution_context": e.get("resolution_context"),
        "node_properties": e.get("node_properties", []),
        "merged_from": e.get("merged_from", [e.get("id")]),
        "members": e.get("members", [e]),
        "merge_history": e.get("merge_history", [])
    }

# ---------------- Build next-iteration entities ----------------
def build_next_entities(resolved_entities: List[Dict],
                        canonical_entities: List[Dict],
                        iteration: int):

    canon_map = {c["canonical_id"]: c for c in canonical_entities if "canonical_id" in c}

    by_canon = {}
    survivors = []

    for e in resolved_entities:
        cid = e.get("canonical_id")
        if cid:
            by_canon.setdefault(cid, []).append(e)
        else:
            survivors.append(e)

    next_entities = []
    n_merges = 0

    # ---- merged groups ----
    for cid, members in by_canon.items():
        canon = canon_map.get(cid, {})
        n_merges += 1

        merged_from = []
        merge_history = []

        for m in members:
            merged_from.extend(m.get("merged_from", [m["id"]]))
            merge_history.extend(m.get("merge_history", []))

        merge_history.append({
            "iteration": iteration,
            "canonical_id": cid,
            "merged_ids": list(set(merged_from))
        })

        # aggregate node properties
        props = []
        seen = set()
        for m in members:
            for p in m.get("node_properties", []) or []:
                k = (p.get("prop_name"), p.get("prop_value"))
                if k not in seen:
                    seen.add(k)
                    props.append(p)

        merged_entity = {
            "id": cid,
            "entity_name": canon.get("canonical_name", members[0]["entity_name"]),
            "entity_description": canon.get("canonical_description", members[0].get("entity_description")),
            "entity_type_hint": canon.get("canonical_type", members[0].get("entity_type_hint")),
            "confidence_score": max(
                [m.get("confidence_score") for m in members if m.get("confidence_score") is not None],
                default=None
            ),
            "resolution_context": canon.get("rationale", ""),
            "node_properties": props,
            "merged_from": sorted(set(merged_from)),
            "members": members,
            "merge_history": merge_history
        }

        next_entities.append(normalize_entity_schema(merged_entity))

    # ---- unchanged entities ----
    for e in survivors:
        e2 = dict(e)
        e2.setdefault("merged_from", [e["id"]])
        e2.setdefault("members", [e])
        e2.setdefault("merge_history", [])
        next_entities.append(normalize_entity_schema(e2))

    return next_entities, n_merges

# ---------------- Main iterative loop ----------------
def iterative_resolution():

    current_input = ENT_RAW_SEED

    # backup seed once
    seed_backup = ITER_DIR / "entities_raw_seed_backup.jsonl"
    if not seed_backup.exists():
        shutil.copy2(ENT_RAW_SEED, seed_backup)

    for it in range(1, DEFAULT_MAX_ITERS + 1):
        print(f"\n================ ITERATION {it} ================")

        # ---- 1) Embedding + clustering ----
        print("[1] Embedding + clustering")
        main_cli(type("Args", (), {
            "entities_in": str(current_input),
            "out_jsonl": str(CLUSTERED_PATH),
            "clusters_summary": str(CLUSTERED_PATH.parent / f"clusters_summary_iter{it}.json"),
            "use_umap": True
        })())

        # ---- 2) Resolution ----
        print("[2] Entity resolution")
        orchestrate()

        # ---- 3) Build next iteration ----
        resolved = load_jsonl(RESOLVED_PATH)
        canon = load_jsonl(CANONICAL_PATH)

        next_entities, n_merges = build_next_entities(resolved, canon, iteration=it)

        print(f"[3] Merges in iteration {it}: {n_merges}")

        # save per-iteration output
        out_path = ITER_DIR / f"entities_iter{it}.jsonl"
        write_jsonl(out_path, next_entities)

        # ---- stopping conditions ----
        if n_merges < MIN_MERGES_TO_CONTINUE:
            print("✔ Stopping: merges below threshold.")
            break

        if it >= DEFAULT_MAX_ITERS:
            print("✔ Stopping: reached max iterations.")
            break

        current_input = out_path
        time.sleep(0.3)

    print("\n=== Iterative resolution COMPLETE ===")
    print(f"All intermediate iterations saved under: {ITER_DIR}")

# ---------------- Run ----------------
if __name__ == "__main__":
    iterative_resolution()





#endregion#? Iterative Ent Res with_per_run_outputs
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Produce_class_input_from_iter K


#!/usr/bin/env python3
"""
Produce a clean JSONL for class-identification with only the requested fields.
"""

import sys, json
from pathlib import Path

# ---------- CONFIG: adjust if needed ----------
input_path = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/iterative_runs/entities_iter3.jsonl")
out_dir = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input")
out_file = out_dir / "cls_input_entities.jsonl"

# ---------- guard against ipykernel injected args ----------
if any(a.startswith("--f=") or a.startswith("--ipykernel") for a in sys.argv[1:]) or "ipykernel" in sys.argv[0]:
    sys.argv = [sys.argv[0]]

# ---------- helpers ----------
def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def synth_id(base_name: str, idx: int):
    safe = (base_name or "no_name").strip().replace(" ", "_")[:40]
    return f"Tmp_{safe}_{idx}"

# ---------- main ----------
def produce_clean_jsonl(inp: Path, outp: Path):
    recs = load_jsonl(inp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    for i, r in enumerate(recs):
        # pick id (prefer top-level id or canonical_id); otherwise synth
        rid = r.get("id") or r.get("canonical_id") or r.get("canonical") or None
        if not rid:
            rid = synth_id(r.get("entity_name"), i)

        # chunk_id may be present as single string or inside members
        chunk_ids = []
        if "chunk_id" in r:
            # can be single or list
            chunk_ids = ensure_list(r.get("chunk_id"))
        else:
            # try to extract from members (if members are dicts with chunk_id)
            members = r.get("members") or []
            for m in ensure_list(members):
                if isinstance(m, dict) and m.get("chunk_id"):
                    chunk_ids.extend(ensure_list(m.get("chunk_id")))
        # dedupe and keep order
        seen = set(); chunk_ids = [c for c in chunk_ids if not (c in seen or seen.add(c))]

        node_props = r.get("node_properties") or []
        # normalize node_properties to list of dicts (best-effort)
        if not isinstance(node_props, list):
            node_props = [node_props]

        cleaned_rec = {
            "id": rid,
            "entity_name": r.get("entity_name") or r.get("canonical_name") or "",
            "entity_description": r.get("entity_description") or r.get("canonical_description") or "",
            "entity_type_hint": r.get("entity_type_hint") or r.get("canonical_type") or r.get("entity_type") or "",
            "confidence_score": r.get("confidence_score") if r.get("confidence_score") is not None else None,
            "resolution_context": r.get("resolution_context") or r.get("text_span") or r.get("context_phrase") or "",
            "flag": r.get("flag") or "entity_raw",
            "chunk_id": chunk_ids,
            "node_properties": node_props
        }
        cleaned.append(cleaned_rec)

    # write out
    with outp.open("w", encoding="utf-8") as fh:
        for rec in cleaned:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned)} records -> {outp}")

if __name__ == "__main__":
    produce_clean_jsonl(input_path, out_file)


#endregion#? Produce_class_input_from_iter K
#?#########################  End  ##########################








#!############################################# Start Chapter ##################################################
#region:#!   Old Ent Res Manual Re Run




#?######################### Start ##########################
#region:#?   Entites after first resolution




#!/usr/bin/env python3
# create_second_run_input.py
"""
Create reduced input for second-pass pipeline.

Outputs in folder: 2nd_run/
 - entities_raw_second_run.jsonl   (one record per canonical or singleton)
 - canonical_members_map.json
 - summary.json
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# ---------- CONFIG ----------
ENT_RES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput")
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_ENT_RAW = OUT_DIR / "entities_raw_second_run.jsonl"
OUT_CANON_MAP = OUT_DIR / "canonical_members_map.json"
OUT_SUMMARY = OUT_DIR / "summary.json"

# ---------- helpers ----------
def load_jsonl(path: Path):
    items = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items

def safe_get(e, k, default=""):
    v = e.get(k)
    if v is None:
        return default
    return v

# ---------- load ----------
print("Loading resolved entities...")
entities = load_jsonl(ENT_RES)
print("Loading canonical entities (if exists)...")
canons = load_jsonl(CANON) if CANON.exists() else []

# build canonical metadata index
canon_meta = {c.get("canonical_id"): c for c in canons}

# group members by canonical_id
members_by_canon = defaultdict(list)
singletons = []
for e in entities:
    cid = e.get("canonical_id")
    if cid:
        members_by_canon[cid].append(e)
    else:
        singletons.append(e)

print(f"Found {len(members_by_canon)} canonical groups and {len(singletons)} singletons (unmerged entities).")

# Build representative records
representatives = []

# 1) canonical representatives
for cid, members in members_by_canon.items():
    meta = canon_meta.get(cid, {})
    canonical_name = meta.get("canonical_name") or safe_get(members[0], "entity_name")
    canonical_desc = meta.get("canonical_description") or safe_get(members[0], "entity_description", "")
    canonical_type = meta.get("canonical_type") or safe_get(members[0], "entity_type_hint", "")
    # Collect up to 5 example member text spans and names for context
    example_spans = []
    example_names = []
    for m in members[:5]:
        if m.get("text_span"):
            example_spans.append(m.get("text_span"))
        example_names.append(m.get("entity_name") or m.get("entity_name_original") or m.get("id"))
    # create representative record (fields align with original entity shape)
    rep = {
        "id": f"CanRep_{cid}",            # synthetic id for the representative
        "canonical_id": cid,
        "entity_name": canonical_name,
        "entity_description": canonical_desc,
        "entity_type_hint": canonical_type,
        "member_count": len(members),
        "example_member_names": example_names,
        "example_member_spans": example_spans,
        # provenance: list of member ids (small sample) and full count
        "member_ids_sample": [m.get("id") for m in members[:10]],
        "_cluster_id": None,
        "_notes": "representative for canonical group"
    }
    representatives.append(rep)

# 2) include singletons as they are (but normalize fields)
single_rep_list = []
for s in singletons:
    rep = {
        "id": s.get("id"),
        "entity_name": safe_get(s, "entity_name"),
        "entity_description": safe_get(s, "entity_description", ""),
        "entity_type_hint": safe_get(s, "entity_type_hint", ""),
        "member_count": 1,
        "example_member_names": [safe_get(s, "entity_name")],
        "example_member_spans": [safe_get(s, "text_span")],
        "member_ids_sample": [s.get("id")],
        "_cluster_id": s.get("_cluster_id"),
        "_notes": "singleton (no canonical_id)"
    }
    single_rep_list.append(rep)

# combined list: canonical reps + singletons
combined = representatives + single_rep_list

# write out entities_raw_second_run.jsonl
with open(OUT_ENT_RAW, "w", encoding="utf-8") as fh:
    for rec in combined:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# write canonical members map for audit
canon_map = {cid: [m.get("id") for m in members] for cid, members in members_by_canon.items()}
with open(OUT_CANON_MAP, "w", encoding="utf-8") as fh:
    json.dump(canon_map, fh, ensure_ascii=False, indent=2)

# write summary
summary = {
    "n_original_entities": len(entities),
    "n_canonical_groups": len(members_by_canon),
    "n_singletons": len(singletons),
    "n_representatives_written": len(combined),
    "path_entities_raw_second_run": str(OUT_ENT_RAW),
    "path_canonical_members_map": str(OUT_CANON_MAP)
}
with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)

print("Wrote:", OUT_ENT_RAW)
print("Wrote:", OUT_CANON_MAP)
print("Wrote:", OUT_SUMMARY)
print("Summary:", summary)






#endregion#? Entites after first resolution
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   RE RUN: Embedding and clustering recognized entities - Forced HDBSCAN - 2nd_run

#!/usr/bin/env python3
"""
embed_and_cluster_entities_force_hdbscan_second_run.py

Ready-to-run script for the SECOND RUN that:
 - Uses the reduced representative input produced in `2nd_run/entities_raw_second_run.jsonl`
 - Forces HDBSCAN clustering (no fixed-K fallback)
 - Optional UMAP reduction
 - Uses BAAI/bge-large-en-v1.5 by default (changeable)
 - Shows progress with tqdm

How to run:
  - Notebook: run the file / import and call main_cli with defaults
  - CLI: python embed_and_cluster_entities_force_hdbscan_second_run.py --entities_in 2nd_run/entities_raw_second_run.jsonl
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`") from e

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Forced HDBSCAN params (as requested)
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # embeddings normalized -> euclidean ~ cosine

# UMAP (optional)
USE_UMAP_DEFAULT = False
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            # fallback dim guess 1024
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        # progress-friendly batching
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state  # (B, T, D)
            pooled = mean_pool(token_embeds, attention_mask)  # (B, D)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")
        # use example_member_spans if available (representative input)
        ctxs.append(safe_text(e, "example_member_spans") or safe_text(e, "text_span") or "")
        types.append(safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or "")
    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs, types = build_field_texts(entities)
    D_ref = None

    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    if emb_name is not None:
        D_ref = emb_name.shape[1]

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    print("[compute] encoding type field ...")
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    if D_ref is None and emb_type is not None:
        D_ref = emb_type.shape[1]

    if D_ref is None:
        raise ValueError("All textual fields empty — cannot embed")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = False):
    print(f"[cluster] HDBSCAN min_cluster_size={min_cluster_size} min_samples={min_samples} metric={metric} use_umap={use_umap}")
    X = embeddings
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
        else:
            # safe n_components: don't exceed n_samples-4
            n_samples = X.shape[0]
            n_components = min(UMAP_N_COMPONENTS, max(2, n_samples - 4))
            print(f"[cluster] running UMAP reduction -> {n_components} dims (safe for n={n_samples})")
            reducer = umap.UMAP(n_components=n_components, n_neighbors=min(UMAP_N_NEIGHBORS, max(2, n_samples-1)),
                                min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
            X = reducer.fit_transform(X)
            print("[cluster] UMAP done, X.shape=", X.shape)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    # summary
    summary = {}
    for idx, lab in enumerate(labels):
        summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

# ------------------ Main flow ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)

    # compute combined embeddings with progress bars per field
    print("Computing combined embeddings (this may take a while)...")
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=args.use_umap if hasattr(args, "use_umap") else USE_UMAP_DEFAULT)

    # diagnostics
    import numpy as _np
    from collections import Counter
    labels_arr = _np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")

# ------------------ Entry point (notebook-friendly) ------------------
if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.argv[0] or "ipython" in sys.argv[0]:
        class Args:
            entities_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput/entities_raw_second_run.jsonl"
            out_jsonl = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Clustering_2nd/entities_clustered_second_run.jsonl"
            clusters_summary = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Clustering_2nd/clusters_summary_second_run.json"
            use_umap = USE_UMAP_DEFAULT
        args = Args()
        print("[main] running in notebook mode with defaults:")
        print(f"  entities_in     = {args.entities_in}")
        print(f"  out_jsonl       = {args.out_jsonl}")
        print(f"  clusters_summary= {args.clusters_summary}")
        print(f"  use_umap        = {args.use_umap}")
        main_cli(args)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--entities_in", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput/entities_raw_second_run.jsonl")
        parser.add_argument("--out_jsonl", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput/entities_clustered_second_run.jsonl")
        parser.add_argument("--clusters_summary", type=str, default="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput/clusters_summary_second_run.json")
        parser.add_argument("--use_umap", action="store_true", help="Enable UMAP reduction before clustering (optional)")
        parsed = parser.parse_args()
        # convert to args-like object
        class ArgsFromCLI:
            entities_in = parsed.entities_in
            out_jsonl = parsed.out_jsonl
            clusters_summary = parsed.clusters_summary
            use_umap = bool(parsed.use_umap)
        args = ArgsFromCLI()
        main_cli(args)


#endregion#? RE RUN: Embedding and clustering recognized entities - Forced HDBSCAN - 2nd_run
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#? Diagnostics for entities_clustered (v2)

# Diagnostics for entities_clustered_second_run.jsonl (v2)
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Clustering_2nd/entities_clustered_second_run.jsonl")  # second-run clustered output
assert IN.exists(), f"{IN} not found — run the second-run clustering first."

ents = []
with open(IN, "r", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            ents.append(json.loads(line))

n = len(ents)
labels = [e.get("_cluster_id", -1) for e in ents]
labels_arr = np.array(labels)
n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
n_noise = int((labels_arr == -1).sum())
print(f"n_entities: {n}, n_clusters (excl -1): {n_clusters}, noise_count: {n_noise} ({n_noise/n*100:.1f}%)")

counts = Counter(labels)
# compute cluster size stats (exclude noise)
cluster_sizes = [sz for lab, sz in counts.items() if lab != -1]
if cluster_sizes:
    print("Cluster size stats: min, median, mean, max =", min(cluster_sizes), np.median(cluster_sizes), np.mean(cluster_sizes), max(cluster_sizes))
else:
    print("No non-noise clusters found.")

# top clusters
top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:15]
print("\nTop 15 clusters (label, size):")
for lab, sz in top:
    print(" ", lab, sz)

# sample members for top 6 clusters
by_label = defaultdict(list)
for i, e in enumerate(ents):
    lab = labels[i]
    name = e.get("entity_name") or e.get("entity_description") or e.get("example_member_names", [None])[0] or f"En_{i}"
    by_label[lab].append(name)

print("\nExamples for top clusters:")
for lab, sz in top[:6]:
    print(f"\nCluster {lab} size={sz}:")
    for v in by_label[lab][:8]:
        print("  -", v)

# small clusters count
small_count = sum(1 for lab, sz in counts.items() if lab != -1 and sz <= 2)
print(f"\nClusters with size <= 2: {small_count}")


#endregion#? Diagnostics for entities_clustered (v2)
#?######################### End ##########################




#?######################### Start ##########################
#region:#?   RE RUN: Entity Resolution - V100 -          V2

# orchestrator_with_chunk_texts_v100_v2.py
"""
Entity resolution orchestrator v100 (V2 defaults).

- Defaults read from 2nd_run/entities_clustered_second_run.jsonl
- Ensures chunk text is loaded and attached to every entity where possible
- LOCAL_HDBSCAN_MIN_CLUSTER_SIZE is set to 2 (for second run)
- Writes final "full" resolved output with all original fields needed for the next step
- Has tqdm progress bars and safe token checks

Usage (notebook):
    from orchestrator_with_chunk_texts_v100_v2 import orchestrate
    orchestrate()

Usage (CLI):
    python orchestrator_with_chunk_texts_v100_v2.py --entities_in 2nd_run/entities_clustered_second_run.jsonl
"""

import os
import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# transformers embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client loader
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Paths & config (V2 defaults) ----------------
CLUSTERED_IN = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Clustering_2nd/entities_clustered_second_run.jsonl")   # default second-run clustered input
CHUNKS_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl")

OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENT_OUT = OUT_DIR / "entities_resolved_v2.jsonl"
CANON_OUT = OUT_DIR / "canonical_entities_v2.jsonl"
LOG_OUT = OUT_DIR / "resolution_log_v2.jsonl"
FULL_OUT = OUT_DIR / "entities_resolved_full_v2.jsonl"   # final output containing all fields needed downstream

# Embedding & weights
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# local HDBSCAN - adjusted for second run (you asked)
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1
LOCAL_HDBSCAN_METRIC = "euclidean"

# UMAP options
LOCAL_USE_UMAP = False
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when local cluster size >= this

# LLM / prompt
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 800

# orchestration thresholds
MAX_CLUSTER_PROMPT = 15
MAX_MEMBERS_PER_PROMPT = 6   # you wanted smaller prompts; set to 6
TRUNC_CHUNK_CHARS = 400
INCLUDE_PREV_CHUNKS = 0

# token safety (rough)
PROMPT_TOKEN_LIMIT = 2200  # char/4 ~ tokens threshold

# ---------------- Utility functions ----------------
def load_chunks(chunks_jsonl_path: Path) -> List[Dict]:
    assert chunks_jsonl_path.exists(), f"Chunks file not found: {chunks_jsonl_path}"
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- HF embedder ----------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 1024))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

def build_field_texts(entities: List[Dict]):
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")
        ctxs.append(safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or "")
        types.append(safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or "")
    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs, types = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx, emb_type):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings; check your entity fields")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined

# ---------------- robust local_subcluster ----------------
def local_subcluster(cluster_entities: List[Dict],
                     entity_id_to_index: Dict[str, int],
                     all_embeddings: np.ndarray,
                     min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
                     use_umap: bool = LOCAL_USE_UMAP,
                     umap_dims: int = UMAP_DIMS):
    from collections import defaultdict
    from sklearn.preprocessing import normalize as _normalize

    idxs = [entity_id_to_index[e["id"]] for e in cluster_entities]
    X = all_embeddings[idxs]
    X = _normalize(X, axis=1)
    n = X.shape[0]

    if n <= 1:
        return {0: list(cluster_entities)} if n == 1 else {-1: []}

    min_cluster_size = min(min_cluster_size, max(2, n))
    if min_samples is None:
        min_samples = max(1, int(min_cluster_size * 0.1))
    else:
        min_samples = min(min_samples, max(1, n - 1))

    X_sub = X
    if use_umap and UMAP_AVAILABLE and n >= UMAP_MIN_SAMPLES_TO_RUN:
        n_components = min(umap_dims, max(2, n - 4))
        try:
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(UMAP_NEIGHBORS, max(2, n - 1)),
                                min_dist=UMAP_MIN_DIST,
                                metric='cosine',
                                random_state=42)
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=LOCAL_HDBSCAN_METRIC,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = defaultdict(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups

# ------------------ LLM helpers ----------------
def call_llm_with_prompt(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = response.choices[0].message.content
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------- Prompt building ----------------
PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
Given the following small cohesive group of candidate entity mentions, decide which ones to MERGE into a single canonical entity, which to MODIFY, and which to KEEP.

Return ONLY a JSON ARRAY. Each element must be one of:
- MergeEntities: {{ "action":"MergeEntities", "entity_ids":[...], "canonical_name":"...", "canonical_description":"...", "canonical_type":"...", "rationale":"..." }}
- ModifyEntity: {{ "action":"ModifyEntity", "entity_id":"...", "new_name":"...", "new_description":"...", "new_type_hint":"...", "rationale":"..." }}
- KeepEntity: {{ "action":"KeepEntity", "entity_id":"...", "rationale":"..." }}

Rules:
- Use ONLY the provided information (name/desc/type_hint/confidence/text_span/chunk_text).
- Be conservative: if unsure, KEEP rather than MERGE.
- Provide short rationale for each action (1-2 sentences).

Group members (id | name | type_hint | confidence | desc | text_span | chunk_text [truncated]):
{members_json}

Return JSON array only (no commentary).
"""

def build_member_with_chunk(m: Dict, chunks_index: Dict[str, Dict]) -> Dict:
    chunk_text = ""
    chunk_id = m.get("chunk_id")
    if chunk_id:
        ch = chunks_index.get(chunk_id)
        if ch:
            ct = ch.get("text", "")
            if INCLUDE_PREV_CHUNKS and isinstance(ch.get("chunk_index_in_section", None), int):
                # optional: include previous chunk(s) - omitted for cost
                pass
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": m.get("text_span"),
        "chunk_text": chunk_text
    }

# ------------------ apply actions ----------------
def apply_actions(members: List[Dict], actions: List[Dict], entities_by_id: Dict[str, Dict],
                  canonical_store: List[Dict], log_entries: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for act in (actions or []):
        typ = act.get("action")
        if typ == "MergeEntities":
            ids = act.get("entity_ids", [])
            canonical_name = act.get("canonical_name")
            canonical_desc = act.get("canonical_description", "")
            canonical_type = act.get("canonical_type", "")
            rationale = act.get("rationale", "")
            can_id = "Can_" + uuid.uuid4().hex[:8]
            canonical = {
                "canonical_id": can_id,
                "canonical_name": canonical_name,
                "canonical_description": canonical_desc,
                "canonical_type": canonical_type,
                "source": "LLM_resolution_v100_v2",
                "rationale": rationale,
                "timestamp": ts
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append({"time": ts, "action": "merge", "canonical_id": can_id, "merged_ids": ids, "rationale": rationale})
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale","")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "modify", "entity_id": eid, "rationale": rationale})
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale","")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "keep", "entity_id": eid, "rationale": rationale})
        else:
            log_entries.append({"time": ts, "action": "unknown", "payload": act})

# ------------------ Orchestration main ----------------
def orchestrate(entities_in: Path = CLUSTERED_IN, chunks_path: Path = CHUNKS_JSONL,
                ent_out: Path = ENT_OUT, canon_out: Path = CANON_OUT,
                log_out: Path = LOG_OUT, full_out: Path = FULL_OUT,
                use_umap: bool = LOCAL_USE_UMAP):
    print("Loading clustered entities from:", entities_in)
    entities = []
    with open(entities_in, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entities.append(json.loads(line))
    print("Loaded entities:", len(entities))

    print("Loading chunks from:", chunks_path)
    chunks = load_chunks(chunks_path) if chunks_path.exists() else []
    print("Loaded chunks:", len(chunks))
    chunks_index = {c.get("id"): c for c in chunks}

    # attach chunk_text for completeness (so final output has it)
    for e in entities:
        cid = e.get("chunk_id")
        if cid and cid in chunks_index:
            e["_chunk_text_truncated"] = (" ".join(chunks_index[cid].get("text","").split()))[:TRUNC_CHUNK_CHARS]
        else:
            e["_chunk_text_truncated"] = ""

    entities_by_id = {e["id"]: e for e in entities}
    entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_embeddings = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined_embeddings.shape)

    by_cluster = defaultdict(list)
    for e in entities:
        by_cluster[e.get("_cluster_id")].append(e)

    canonical_store = []
    log_entries = []

    cluster_ids = sorted([k for k in by_cluster.keys() if k != -1])
    noise_count = len(by_cluster.get(-1, []))
    print("Clusters to resolve (excluding noise):", len(cluster_ids), "noise_count:", noise_count)

    # iterate clusters with tqdm
    with tqdm(cluster_ids, desc="Clusters", unit="cluster") as pbar_clusters:
        for cid in pbar_clusters:
            members = by_cluster[cid]
            size = len(members)
            pbar_clusters.set_postfix(cluster=cid, size=size)

            if size <= MAX_CLUSTER_PROMPT:
                n_prompts = math.ceil(size / MAX_MEMBERS_PER_PROMPT)
                with tqdm(range(n_prompts), desc=f"Cluster {cid} prompts", leave=False, unit="prompt") as pbar_prompts:
                    for i in pbar_prompts:
                        s = i * MAX_MEMBERS_PER_PROMPT
                        chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            for m in chunk:
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = call_llm_with_prompt(prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # local sub-cluster
                subgroups = local_subcluster(members, entity_id_to_index, combined_embeddings,
                                            min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            use_umap=use_umap, umap_dims=UMAP_DIMS)
                sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                with tqdm(sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub") as pbar_subs:
                    for sublab, submembers in pbar_subs:
                        subsize = len(submembers)
                        pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                        if sublab == -1:
                            for m in submembers:
                                m["resolved_action"] = "kept_noise_local"
                                m["resolution_rationale"] = "Local-subcluster noise preserved"
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"keep_noise_local", "entity_id": m["id"], "cluster": cid})
                            continue
                        n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                        with tqdm(range(n_prompts), desc=f"Sub {sublab} prompts", leave=False, unit="prompt") as pbar_sub_prompts:
                            for i in pbar_sub_prompts:
                                s = i * MAX_MEMBERS_PER_PROMPT
                                chunk = submembers[s:s+MAX_MEMBERS_PER_PROMPT]
                                payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                est_tokens = max(1, int(len(prompt) / 4))
                                pbar_sub_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                                if est_tokens > PROMPT_TOKEN_LIMIT:
                                    for m in chunk:
                                        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                            "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                            "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                        m["resolved_action"] = "kept_skipped_prompt"
                                        m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                    continue
                                llm_out = call_llm_with_prompt(prompt)
                                actions = extract_json_array(llm_out)
                                if actions is None:
                                    actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                                apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"keep_noise_global", "entity_id": ent["id"]})

    # Final: ensure full output contains everything we might need downstream:
    # include original keys plus added resolution fields and truncated chunk text
    final_entities = list(entities_by_id.values())

    # Write standard outputs
    with open(ent_out, "w", encoding="utf-8") as fh:
        for e in final_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(canon_out, "w", encoding="utf-8") as fh:
        for c in canonical_store:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(log_out, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    # Write FULL output used by next pipeline stage:
    # For each entity, include: id, entity_name, entity_description, entity_type_hint, confidence_score,
    # chunk_id, chunk_text_truncated, ref_index, chunk_index_in_section, ref_title, original_raw_llm_if_any, canonical_id, resolved_action
    with open(full_out, "w", encoding="utf-8") as fh:
        for e in final_entities:
            full = {
                "id": e.get("id"),
                "entity_name": e.get("entity_name"),
                "entity_description": e.get("entity_description"),
                "entity_type_hint": e.get("entity_type_hint"),
                "confidence_score": e.get("confidence_score"),
                "chunk_id": e.get("chunk_id"),
                "chunk_text_truncated": e.get("_chunk_text_truncated", ""),
                "ref_index": e.get("ref_index"),
                "chunk_index_in_section": e.get("chunk_index_in_section"),
                "ref_title": e.get("ref_title"),
                "text_span": e.get("text_span"),
                "used_context_excerpt": e.get("used_context_excerpt"),
                "canonical_id": e.get("canonical_id"),
                "resolved_action": e.get("resolved_action"),
                "resolution_rationale": e.get("resolution_rationale"),
                "_raw_llm": e.get("_raw_llm")
            }
            fh.write(json.dumps(full, ensure_ascii=False) + "\n")

    print("\nResolution finished. Wrote:", ent_out, canon_out, log_out, full_out)

# ------------------ Notebook/CLI entry ----------------
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--entities_in", type=str, default=str(CLUSTERED_IN))
    parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
    parser.add_argument("--out_entities", type=str, default=str(ENT_OUT))
    parser.add_argument("--canon_out", type=str, default=str(CANON_OUT))
    parser.add_argument("--log_out", type=str, default=str(LOG_OUT))
    parser.add_argument("--full_out", type=str, default=str(FULL_OUT))
    parser.add_argument("--use_umap", action="store_true", help="Enable local UMAP inside sub-clustering")
    parser.add_argument("--prompt_token_limit", type=int, default=PROMPT_TOKEN_LIMIT)
    parser.add_argument("--max_members_per_prompt", type=int, default=MAX_MEMBERS_PER_PROMPT)
    args = parser.parse_args()

    CLUSTERED_IN = Path(args.entities_in)
    CHUNKS_JSONL = Path(args.chunks)
    ENT_OUT = Path(args.out_entities)
    CANON_OUT = Path(args.canon_out)
    LOG_OUT = Path(args.log_out)
    FULL_OUT = Path(args.full_out)
    PROMPT_TOKEN_LIMIT = args.prompt_token_limit
    MAX_MEMBERS_PER_PROMPT = args.max_members_per_prompt
    if args.use_umap:
        LOCAL_USE_UMAP = True

    orchestrate(entities_in=CLUSTERED_IN, chunks_path=CHUNKS_JSONL,
                ent_out=ENT_OUT, canon_out=CANON_OUT, log_out=LOG_OUT, full_out=FULL_OUT,
                use_umap=LOCAL_USE_UMAP)



#endregion#? RE RUN: Entity Resolution - V100 -          V2
#?#########################  End  ##########################








#?######################### Start ##########################
#region:#?   Analyze_entity_resolution        -    V2

#!/usr/bin/env python3
"""
analyze_entity_resolution_v2.py

Analysis for V2 entity-resolution outputs. Place this script next to your repo root
and run. It reads resolved/clustered/canonical files from 2nd_run/ by default and
writes a comprehensive analysis into 2nd_run/entResAnalysis/.

Outputs (examples):
 - merged_groups.json        : canonical_id -> list of full member entity dicts
 - merged_groups.csv         : flat CSV with one row per merged member
 - canonical_summary.csv     : per-canonical metadata and examples
 - actions_summary.json      : counts per resolved_action
 - type_distribution.csv     : distribution of entity_type_hint for merged vs unmerged
 - merges_hist.png           : histogram of canonical group sizes
 - actions_pie.png           : pie chart of top actions
 - top50_canonical.csv       : top 50 canonical groups
 - canonical_to_members_sample.json : simple mapping for quick inspection
 - unmerged_entities.jsonl   : entities not assigned to any canonical (kept singletons)
 - raw_to_v2_comparison.csv  : for each original raw entity (if provided) whether it was merged and where
 - per_cluster_stats.csv     : stats grouped by coarse _cluster_id (if present)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# ---------------- Config: file locations (update if needed) ----------------
BASE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd")
BASE.mkdir(parents=True, exist_ok=True)

# Expected inputs (V2)
ENT_RES_FILE = BASE / "entities_resolved_v2.jsonl"        # output of orchestrator V2 (full entities with canonical_id etc.)
CANON_FILE = BASE / "canonical_entities_v2.jsonl"         # canonical records created by orchestrator V2
CLUSTERED_FILE = BASE / "entities_clustered_second_run.jsonl"  # clustered input used for orchestrator (optional but useful)
RAW_FILE = BASE / "entities_raw_second_run.jsonl"         # original raw entities from 2nd run (optional; helpful for comparisons)

# Output analysis folder
OUT_DIR = BASE / "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_2nd_Analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                data.append(json.loads(ln))
            except Exception as e:
                print(f"[warn] skipping jsonl line in {path}: {e}")
    return data

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------------- Load inputs ----------------
print("Loading inputs...")
entities_res = load_jsonl(ENT_RES_FILE)
canonals = load_jsonl(CANON_FILE)
clustered = load_jsonl(CLUSTERED_FILE)
raw_entities = load_jsonl(RAW_FILE)

print(f"Loaded: resolved={len(entities_res)} canonical={len(canonals)} clustered={len(clustered)} raw={len(raw_entities)}")

# Build dicts
entities_by_id = {e.get("id"): e for e in entities_res}
canon_by_id = {c.get("canonical_id"): c for c in canonals}

# ---------------- Build merged groups & unmerged ----------------
merged = defaultdict(list)   # canonical_id -> list of member entities
unmerged = []                # entities without canonical_id (singletons kept)

for e in entities_res:
    cid = e.get("canonical_id")
    if cid:
        merged[cid].append(e)
    else:
        unmerged.append(e)

# Save merged_groups.json (full objects)
save_json({k: v for k, v in merged.items()}, OUT_DIR / "merged_groups.json")

# ---------------- Flatten merged groups to CSV ----------------
csv_fields = [
    "canonical_id", "canonical_name",
    "member_id", "member_name", "member_desc", "member_type",
    "confidence_score", "_cluster_id", "resolved_action", "resolution_rationale", "source_ref"
]
rows = []
for cid, members in merged.items():
    canon_name = canon_by_id.get(cid, {}).get("canonical_name", "")
    for m in members:
        rows.append({
            "canonical_id": cid,
            "canonical_name": canon_name,
            "member_id": m.get("id"),
            "member_name": m.get("entity_name"),
            "member_desc": m.get("entity_description"),
            "member_type": m.get("entity_type_hint"),
            "confidence_score": m.get("confidence_score"),
            "_cluster_id": m.get("_cluster_id"),
            "resolved_action": m.get("resolved_action"),
            "resolution_rationale": m.get("resolution_rationale",""),
            "source_ref": m.get("chunk_id") or m.get("ref_index") or ""
        })

write_csv_rows(OUT_DIR / "merged_groups.csv", csv_fields, rows)

# ---------------- canonical_summary.csv ----------------
canon_rows = []
for cid, members in merged.items():
    cmeta = canon_by_id.get(cid, {})
    example_names = " | ".join([m.get("entity_name","") for m in members[:6]])
    canon_rows.append({
        "canonical_id": cid,
        "canonical_name": cmeta.get("canonical_name",""),
        "canonical_type": cmeta.get("canonical_type",""),
        "n_members": len(members),
        "example_members": example_names,
        "rationale": cmeta.get("rationale","")
    })
canon_df = pd.DataFrame(canon_rows).sort_values("n_members", ascending=False)
canon_df.to_csv(OUT_DIR / "canonical_summary.csv", index=False)

# ---------------- actions_summary.json ----------------
action_counts = Counter([e.get("resolved_action","<none>") for e in entities_res])
save_json(dict(action_counts), OUT_DIR / "actions_summary.json")

# ---------------- type distribution (merged vs unmerged) ----------------
def type_counter(list_of_entities: List[Dict[str,Any]]):
    c = Counter()
    for e in list_of_entities:
        t = e.get("entity_type_hint") or "<unknown>"
        c[t] += 1
    return c

merged_types = type_counter([m for members in merged.values() for m in members])
unmerged_types = type_counter(unmerged)
all_types = sorted(set(list(merged_types.keys()) + list(unmerged_types.keys())))

with open(OUT_DIR / "type_distribution.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["type", "merged_count", "unmerged_count"])
    for t in all_types:
        w.writerow([t, merged_types.get(t,0), unmerged_types.get(t,0)])

# ---------------- Quick stats & charts ----------------
# 1) Histogram: canonical group sizes
sizes = [len(members) for members in merged.values()]
if sizes:
    plt.figure(figsize=(6,4))
    plt.hist(sizes, bins=range(1, max(sizes)+2), edgecolor='black', align='left')
    plt.title("Distribution of canonical group sizes (# members)")
    plt.xlabel("members per canonical entity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "merges_hist.png", dpi=150)
    plt.close()

# 2) Pie: top actions
actions_df = pd.DataFrame(list(action_counts.items()), columns=["action","count"]).sort_values("count", ascending=False)
plt.figure(figsize=(6,6))
top_actions = actions_df.head(8)
plt.pie(top_actions["count"], labels=top_actions["action"], autopct="%1.1f%%", startangle=140)
plt.title("Top resolved_action distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "actions_pie.png", dpi=150)
plt.close()

# 3) Top canonical groups CSV (top 50)
canon_df.head(50).to_csv(OUT_DIR / "top50_canonical.csv", index=False)

# 4) Simple mapping sample
simple_map = {cid: [m.get("entity_name") for m in members[:20]] for cid,members in merged.items()}
save_json(simple_map, OUT_DIR / "canonical_to_members_sample.json")

# 5) Save unmerged entities for manual review
with open(OUT_DIR / "unmerged_entities.jsonl", "w", encoding="utf-8") as fh:
    for e in unmerged:
        fh.write(json.dumps(e, ensure_ascii=False) + "\n")

# ---------------- Per-cluster summary (if _cluster_id exists) ----------------
per_cluster = defaultdict(list)
for e in entities_res:
    lab = e.get("_cluster_id", None)
    per_cluster[lab].append(e)
per_cluster_rows = []
for lab, members in sorted(per_cluster.items(), key=lambda x: (x[0] is None, x[0])):
    n_total = len(members)
    n_merged = sum(1 for m in members if m.get("canonical_id"))
    n_kept = sum(1 for m in members if not m.get("canonical_id"))
    action_counts_cluster = Counter([m.get("resolved_action","<none>") for m in members])
    per_cluster_rows.append({
        "_cluster_id": lab,
        "n_members": n_total,
        "n_merged": n_merged,
        "n_kept": n_kept,
        "top_actions": ";".join(f"{k}:{v}" for k,v in action_counts_cluster.most_common(5))
    })
per_cluster_df = pd.DataFrame(per_cluster_rows)
per_cluster_df.to_csv(OUT_DIR / "per_cluster_stats.csv", index=False)

# ---------------- Raw -> V2 comparison (if raw file provided) ----------------
# We want a table where each raw entity id maps to final canonical_id (if any),
# final entity_id (if modified), resolution action, cluster id, etc.
if raw_entities:
    # Build a lookup from member names or unique ids: prefer id matching
    # raw entities often had same 'id' as later records; check both scenarios.
    raw_map = {r.get("id"): r for r in raw_entities if r.get("id")}
    rows_cmp = []
    for raw in raw_entities:
        rid = raw.get("id")
        # Find resolved entity: try exact id match; else try to match by name
        resolved = entities_by_id.get(rid)
        if resolved is None:
            # fallback: try to match by name (casefold)
            raw_name = (raw.get("entity_name") or "").strip().casefold()
            candidates = [e for e in entities_res if (e.get("entity_name") or "").strip().casefold() == raw_name]
            resolved = candidates[0] if candidates else None
        rows_cmp.append({
            "raw_id": rid,
            "raw_name": raw.get("entity_name"),
            "raw_desc": raw.get("entity_description"),
            "resolved_exists": bool(resolved),
            "resolved_id": resolved.get("id") if resolved else "",
            "canonical_id": resolved.get("canonical_id") if resolved else "",
            "resolved_action": resolved.get("resolved_action") if resolved else "",
            "_cluster_id": resolved.get("_cluster_id") if resolved else "",
            "notes": ""
        })
    cmp_df = pd.DataFrame(rows_cmp)
    cmp_df.to_csv(OUT_DIR / "raw_to_v2_comparison.csv", index=False)

# ---------------- Some additional helpful views ----------------
# 1) Ranked merges by size
ranked = sorted(((cid, len(members)) for cid,members in merged.items()), key=lambda x: -x[1])
with open(OUT_DIR / "ranked_merges.txt", "w", encoding="utf-8") as fh:
    for cid, sz in ranked:
        fh.write(f"{cid}\t{sz}\t{canon_by_id.get(cid,{}).get('canonical_name','')}\n")

# 2) Save canonical entities file copy (for convenience)
if canonals:
    save_json(canonals, OUT_DIR / "canonical_entities_v2_copy.json")

# 3) Summary print
print("\n=== Analysis summary ===")
print("analysis folder:", OUT_DIR)
print("total resolved entities:", len(entities_res))
print("canonical groups (merged):", len(merged))
print("unmerged singletons:", len(unmerged))
print("canonical summary top-10:")
if not canon_df.empty:
    print(canon_df.head(10).to_string(index=False))
print("\nTop action counts:")
for a,cnt in action_counts.most_common(10):
    print(" ", a, cnt)

# 4) List produced files
print("\nFiles produced:")
for p in sorted(OUT_DIR.iterdir()):
    print(" ", p.name)

print("\nDone.")


#endregion#? Analyze_entity_resolution        -    V2
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   Entites after second run -    V2


#!/usr/bin/env python3
# create_second_run_input_v2.py
"""
Create reduced input (representative entities) after the SECOND resolution run.

Outputs in folder: 2nd_run/
 - entities_raw_second_run.jsonl   (one record per canonical representative OR singleton final entity)
 - canonical_members_map_v2.json   (canonical_id -> full list of member ids)
 - summary_v2.json                 (counts and paths)

Notes:
 - The script will try to locate the SECOND-RUN resolved entities file from a few likely paths.
 - If you know the exact path, set the RESOLVED_V2_PATH variable or pass it via environment before running.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import sys
import os


# ----------------- Config / candidate input paths -----------------
# Set these to your desired locations
RESOLVED_BASE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd")
RESOLVED_BASE.mkdir(exist_ok=True, parents=True)

OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_2nd_Ouput")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# candidate locations (searched in order) for resolved second-run
CANDIDATES = [
    RESOLVED_BASE / "entities_resolved.jsonl",
    RESOLVED_BASE / "entities_resolved_v2.jsonl",
    RESOLVED_BASE / "entities_resolved_second_run.jsonl",
    RESOLVED_BASE / "entities_resolved_second_run_final.jsonl",
    Path("entities_resolved_v2.jsonl"),
    Path("entities_resolved_second_run.jsonl"),
    Path("entities_resolved.jsonl"),
]

# canonical records (optional) - absolute or relative paths (no incorrect BASE_DIR / "/abs" concatenation)
CANDIDON_CANON = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/canonical_entities_v2.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl"),
    Path("canonical_entities_v2.jsonl"),
    Path("canonical_entities.jsonl"),
]

# outputs
OUT_ENT_RAW = OUT_DIR / "entities_raw_second_run.jsonl"
OUT_CANON_MAP = OUT_DIR / "canonical_members_map_v2.json"
OUT_SUMMARY = OUT_DIR / "summary_v2.json"

# optional: raw/clustered inputs from this run (if present) to enrich representative
RAW_CANDIDATES = [
    RESOLVED_BASE / "entities_raw_second_run_source.jsonl",
    RESOLVED_BASE / "entities_raw_second_run.jsonl",
    RESOLVED_BASE / "entities_raw.jsonl",
    Path("entities_raw_second_run.jsonl"),
]


#!tree /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities


# ----------------- helpers -----------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception as e:
                print(f"[warn] skipping invalid json line in {path}: {e}")
    return items

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

# ----------------- locate inputs -----------------
resolved_v2_path = None
for p in CANDIDATES:
    if p.exists():
        resolved_v2_path = p
        break

if resolved_v2_path is None:
    print("ERROR: Could not find a second-run resolved entities file in any of the candidate locations:")
    for p in CANDIDATES:
        print("  -", p)
    sys.exit(1)

print("Using resolved entities (second-run):", resolved_v2_path)

# optional canonical file
canon_path = None
for p in CANDIDON_CANON:
    if p.exists():
        canon_path = p
        break
if canon_path:
    print("Using canonical records:", canon_path)
else:
    print("No canonical records file found (canonical records optional).")

# optional raw/clustered inputs for enrichment (best-effort)
raw_source_path = None
for p in RAW_CANDIDATES:
    if p.exists():
        raw_source_path = p
        break
if raw_source_path:
    print("Using raw/clustered source (for enrichment) from:", raw_source_path)
else:
    print("No raw/clustered source file found in candidates (enrichment skipped if absent).")

# ----------------- load data -----------------
resolved = load_jsonl(resolved_v2_path)
canon_records = load_jsonl(canon_path) if canon_path else []
raw_source = load_jsonl(raw_source_path) if raw_source_path else []

print(f"Loaded: resolved={len(resolved)}, canonical_records={len(canon_records)}, raw_source={len(raw_source)}")

# build maps for quick lookup
resolved_map = {r.get("id"): r for r in resolved if r.get("id")}
raw_map = {r.get("id"): r for r in raw_source if r.get("id")}

# canonical_members from canonical records (if present) - prefer these lists when available
canonical_members = defaultdict(list)  # canonical_id -> list(member_ids)
for c in canon_records:
    cid = c.get("canonical_id")
    # canonical record may store "members", "member_ids", "merged_ids" or similar
    members = c.get("members") or c.get("member_ids") or c.get("merged_ids") or c.get("members_ids") or []
    if cid:
        if isinstance(members, list):
            canonical_members[cid].extend(members)
        else:
            # sometimes members may be stored as comma-separated string
            if isinstance(members, str) and members.strip():
                canonical_members[cid].extend([x.strip() for x in members.split(",") if x.strip()])

# Build reverse map: member_id -> canonical_id (if known from canonical records)
member_to_canon = {}
for cid, mids in canonical_members.items():
    for m in mids:
        member_to_canon[m] = cid

# ----------------- build representatives -----------------
representatives = []   # final rows to write (one per final entity)
canon_map_out = {}     # canonical_id -> full member ids (unioned)

# First, group resolved rows that share the same canonical_id (some resolved rows may still hold canonical_id)
by_canon = defaultdict(list)
singletons = []

for r in resolved:
    cid = r.get("canonical_id")
    if cid:
        by_canon[cid].append(r)
    else:
        # Entities without canonical_id: treat as singleton representative but keep member history if present
        singletons.append(r)

# For canonical groups: build a single representative per canonical_id
for cid, members in by_canon.items():
    # gather member ids (prefer canonical_records list if available, else union of member fields in resolved members)
    mids = []
    if canonical_members.get(cid):
        mids = list(dict.fromkeys(canonical_members[cid]))  # dedupe preserve order
    else:
        # collect from resolved members (fields like merged_from, member_ids, merged_ids, members)
        for m in members:
            for fld in ("member_ids","merged_ids","merged_from","members"):
                val = m.get(fld)
                if isinstance(val, list):
                    for mm in val:
                        if mm and mm not in mids:
                            mids.append(mm)
            # also include this resolved row's id
            rid = m.get("id")
            if rid and rid not in mids:
                mids.append(rid)
    # as a final fallback, ensure member ids include the resolved rows' ids
    for m in members:
        if m.get("id") and m.get("id") not in mids:
            mids.append(m.get("id"))

    canon_map_out[cid] = mids

    # try to infer canonical metadata (name/desc/type) from canonical record if present, else from highest-confidence member
    canon_meta = {}
    for rec in canon_records:
        if rec.get("canonical_id") == cid:
            canon_meta = rec
            break

    # choose canonical_name/desc/type: prefer canon_meta, else choose member with highest confidence
    canonical_name = canon_meta.get("canonical_name") if canon_meta.get("canonical_name") else None
    canonical_desc = canon_meta.get("canonical_description") if canon_meta.get("canonical_description") else None
    canonical_type = canon_meta.get("canonical_type") if canon_meta.get("canonical_type") else None

    if not canonical_name:
        # pick member with highest confidence (if any)
        sorted_members = sorted(members, key=lambda x: float(x.get("confidence_score") or 0.0), reverse=True)
        if sorted_members:
            canonical_name = canonical_name or sorted_members[0].get("entity_name")
            canonical_desc = canonical_desc or sorted_members[0].get("entity_description")
            canonical_type = canonical_type or sorted_members[0].get("entity_type_hint")

    # Build representative record (keep provenance info)
    rep = {
        "id": f"CanRep_{cid}",
        "canonical_id": cid,
        "entity_name": canonical_name,
        "entity_description": canonical_desc,
        "entity_type_hint": canonical_type,
        "member_count": len(mids),
        "member_ids": mids,
        "member_sample": [],
        "member_details_sample": [],
        "_notes": "representative for canonical group (second-run)"
    }

    # attach up to 8 member samples with best available fields
    samp = []
    samp_details = []
    for mid in mids[:8]:
        src = resolved_map.get(mid) or raw_map.get(mid) or {}
        samp.append(src.get("entity_name") or mid)
        samp_details.append({
            "id": mid,
            "name": src.get("entity_name"),
            "description": src.get("entity_description"),
            "type": src.get("entity_type_hint"),
            "confidence": src.get("confidence_score"),
            "chunk_id": src.get("chunk_id") or src.get("source_chunk_id") or src.get("ref_index")
        })
    rep["member_sample"] = samp
    rep["member_details_sample"] = samp_details

    representatives.append(rep)

# Now singletons: normalize and include metadata + any merged history fields
for s in singletons:
    # collect any member lists embedded in the singleton (if it was previously a rep with member_ids)
    mids = []
    for fld in ("member_ids","merged_ids","merged_from","members"):
        val = s.get(fld)
        if isinstance(val, list):
            for mm in val:
                if mm and mm not in mids:
                    mids.append(mm)
    if not mids:
        mids = [s.get("id")]

    rep = {
        "id": s.get("id"),
        "canonical_id": s.get("canonical_id") or None,
        "entity_name": s.get("entity_name"),
        "entity_description": s.get("entity_description"),
        "entity_type_hint": s.get("entity_type_hint"),
        "member_count": len(mids),
        "member_ids": mids,
        "member_sample": [s.get("entity_name")],
        "member_details_sample": [{
            "id": s.get("id"),
            "name": s.get("entity_name"),
            "description": s.get("entity_description"),
            "type": s.get("entity_type_hint"),
            "confidence": s.get("confidence_score"),
            "chunk_id": s.get("chunk_id") or s.get("source_chunk_id") or s.get("ref_index")
        }],
        "_notes": "singleton (no canonical group in second-run)"
    }
    representatives.append(rep)

# final ordering: prefer canonical reps first then singletons
# but keep as is (representatives already contains canonical then singletons)

# ----------------- write outputs -----------------
write_jsonl(OUT_ENT_RAW, representatives)
write_json(OUT_CANON_MAP, canon_map_out)

summary = {
    "resolved_v2_path": str(resolved_v2_path),
    "resolved_v2_count": len(resolved),
    "n_canonical_groups": len(by_canon),
    "n_singletons": len(singletons),
    "n_representatives_written": len(representatives),
    "path_entities_raw_second_run": str(OUT_ENT_RAW),
    "path_canonical_members_map_v2": str(OUT_CANON_MAP)
}
write_json(OUT_SUMMARY, summary)

print("Wrote representatives for second-run to:", OUT_ENT_RAW)
print("Wrote canonical members map to:", OUT_CANON_MAP)
print("Wrote summary to:", OUT_SUMMARY)
print("Summary:", summary)


#endregion#? Entites after second run -    V2
#?#########################  End  ##########################












#?######################### Start ##########################
#region:#?   Final Analysis


#!/usr/bin/env python3
"""
create_final_comp_analysis.py

Produces a consolidated analysis folder "FinalCompAnalysis" inside:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Overall_After_Two_Run/FinalCompAnalysis

What it does:
- Locates the best-guess input files produced across your two resolution runs
  (resolved entities, canonical records, clustered inputs, raw inputs, analysis outputs)
- Builds a master JSONL (and JSON) file with one row per final entity after second-run resolution.
  Each entity row contains: id, entity_name, entity_description, entity_type_hint, confidence_score,
  chunk_ids (list), member_ids (list of merged ids across runs), canonical_id (if any),
  resolved_action, resolution_rationale, _cluster_id, chunk_text_truncated, ref_index, chunk_index_in_section,
  ref_title, _raw_llm (if present), and any other fields present in the entity record.
- Produces CSV summaries and simple plots (merges histogram, actions pie).
- Copies available diagnostic/analysis files into the FinalCompAnalysis folder.
- Writes a README.md and a short markdown summary report.

Notes:
- The script is defensive: it tries multiple candidate paths (based on your repo files)
  and continues even if some files are missing or inconsistent.
- Requires: pandas, matplotlib (install with pip if missing).
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import itertools
import logging
import csv

# Optional plotting libs
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as e:
    pd = None
    plt = None

# ---------- Config: target output folder ----------
BASE_TARGET = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Overall_After_Two_Run")
FINAL_DIR = BASE_TARGET / "FinalCompAnalysis"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = FINAL_DIR / "create_final_comp_analysis.log"
logging.basicConfig(level=logging.INFO, filename=str(LOG_PATH), filemode="w",
                    format="%(asctime)s %(levelname)s: %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Starting FinalCompAnalysis builder")
logging.info(f"Output folder: {FINAL_DIR}")

# ---------- Candidate input paths (from repo context) ----------
# We'll try to load from these candidate paths in order; the first matching file is used.
CANDIDATE_RESOLVED = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/entities_resolved_v2.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/entities_resolved.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/entities_resolved_second_run.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/entities_resolved_second_run_final.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl"),
]

CANDIDATE_CANON = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/canonical_entities_v2.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd/canonical_entities.jsonl"),
]

CANDIDATE_CLUSTERED = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Clustering_2nd/entities_clustered_second_run.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl"),
]

CANDIDATE_RAW_REPS = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_2nd_Ouput/entities_raw_second_run.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Ouput/entities_raw_second_run.jsonl"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_raw.jsonl"),
]

CANDIDATE_ANALYSIS_DIRS = [
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_2nd_Analysis"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_1st_Analysis"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st"),
    Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_2nd/Ent_Resolved_2nd"),
]

# also look for any json/csv/png in the Entities tree (as fallback copy)
SEARCH_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities")

# ---------- helpers ----------
def find_first_existing(paths):
    for p in paths:
        if p.exists():
            logging.info(f"Found: {p}")
            return p
    logging.warning("No candidate file found in provided list.")
    return None

def load_jsonl(path):
    items = []
    if not path or not path.exists():
        logging.warning(f"load_jsonl: missing {path}")
        return items
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception as e:
                logging.warning(f"Skipping invalid json line in {path}: {e}")
    logging.info(f"Loaded {len(items)} records from {path.name}")
    return items

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"Wrote JSONL: {path} ({len(rows)} rows)")

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    logging.info(f"Wrote JSON: {path}")

# copy file safely
def safe_copy(src: Path, dst_dir: Path):
    try:
        if not src.exists():
            return None
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        logging.info(f"Copied {src} -> {dst}")
        return dst
    except Exception as e:
        logging.warning(f"Failed to copy {src}: {e}")
        return None

# ---------- Locate inputs ----------
resolved_path = find_first_existing(CANDIDATE_RESOLVED)
canon_path = find_first_existing(CANDIDATE_CANON)
clustered_path = find_first_existing(CANDIDATE_CLUSTERED)
raw_rep_path = find_first_existing(CANDIDATE_RAW_REPS)

# Also attempt to find 'entities_resolved_full_v2.jsonl' or similar "full" outputs referenced in your code:
additional_full_candidates = list(SEARCH_ROOT.glob("**/*entities_resolved*_full*.jsonl")) + list(SEARCH_ROOT.glob("**/*entities_resolved_v2*.jsonl"))
if additional_full_candidates:
    # prefer ones under Ent_Resolved_2nd
    additional_full = sorted(additional_full_candidates, key=lambda p: ("Ent_Resolved_2nd" in str(p), str(p)), reverse=True)
    if additional_full:
        chosen = additional_full[0]
        if resolved_path is None:
            resolved_path = chosen
            logging.info(f"Selected additional resolved candidate: {chosen}")

# Load files if present
resolved = load_jsonl(resolved_path) if resolved_path else []
canons = load_jsonl(canon_path) if canon_path else []
clustered = load_jsonl(clustered_path) if clustered_path else []
raw_reps = load_jsonl(raw_rep_path) if raw_rep_path else []

# also try to load first-run resolved/canon if available to recover member history
first_run_resolved_candidates = list(SEARCH_ROOT.glob("**/Ent_Resolved_1st/**/entities_resolved*.jsonl"))
first_run_canon_candidates = list(SEARCH_ROOT.glob("**/Ent_Resolved_1st/**/canonical_entities*.jsonl"))
first_run_resolved = load_jsonl(first_run_resolved_candidates[0]) if first_run_resolved_candidates else []
first_run_canons = load_jsonl(first_run_canon_candidates[0]) if first_run_canon_candidates else []

# ---------- Build maps to aggregate member histories ----------
# We'll try to aggregate member ids that were merged across runs by checking:
# - canonical records 'member_ids' / 'merged_ids' fields
# - resolved entity fields that include 'member_ids', 'merged_ids', 'merged_from', etc.
member_to_canonical = {}
canonical_members = defaultdict(list)

def aggregate_from_canonical_records(canon_list):
    for c in canon_list:
        cid = c.get("canonical_id") or c.get("canonical") or c.get("id")
        if not cid:
            continue
        # possible member fields
        members = []
        for fld in ("members","member_ids","merged_ids","merged_from","merged"):
            val = c.get(fld)
            if isinstance(val, list):
                members.extend(val)
            elif isinstance(val, str) and val.strip():
                # try to split by comma or space
                parts = [x.strip() for x in val.split(",") if x.strip()]
                if parts:
                    members.extend(parts)
        # also some canonical records may embed 'members' as objects
        if not members:
            # try keys like 'members_details'
            md = c.get("members_details") or c.get("members_info")
            if isinstance(md, list):
                for m in md:
                    mid = m.get("id") or m.get("member_id")
                    if mid:
                        members.append(mid)
        # dedupe and add
        if members:
            seen = []
            for m in members:
                if m and m not in seen:
                    seen.append(m)
            canonical_members[cid].extend(seen)
            for m in seen:
                member_to_canonical[m] = cid

# aggregate from canonical files (both runs)
aggregate_from_canonical_records(canons)
aggregate_from_canonical_records(first_run_canons)

# Also scan resolved entities for any member lists embedded per-entity and populate canonical_members mapping
def aggregate_from_resolved(resolved_list):
    for r in resolved_list:
        cid = r.get("canonical_id")
        if cid:
            # check fields for explicit member lists
            for fld in ("member_ids","merged_ids","merged_from","members"):
                val = r.get(fld)
                if isinstance(val, list):
                    for m in val:
                        if m and m not in canonical_members[cid]:
                            canonical_members[cid].append(m)
                            member_to_canonical[m] = cid
            # if not present, include this resolved record id as member
            if r.get("id") and r.get("id") not in canonical_members[cid]:
                canonical_members[cid].append(r.get("id"))
                member_to_canonical[r.get("id")] = cid

aggregate_from_resolved(resolved)
aggregate_from_resolved(first_run_resolved)

logging.info(f"Aggregated {len(canonical_members)} canonical groups from available records")

# ---------- Build master entity list (one row per *final* entity after second-run) ----------
# Strategy:
# - If 'resolved' contains final entities (it should), use those as primary rows
# - For each final entity, collect:
#     * all known member_ids (from entity fields and canonical_members)
#     * chunk_ids found in member records (search in resolved + first-run + raw reps)
#     * best available metadata (name/desc/type/confidence) preferring canonical metadata then highest-confidence member
# - If resolved is empty, fall back to 'raw_reps' or first_run_resolved etc.

# helper to collect member ids for a resolved entity
def collect_member_ids(entity):
    mids = set()
    # fields on the entity itself
    for fld in ("member_ids","merged_ids","merged_from","members","member_ids_sample"):
        val = entity.get(fld)
        if isinstance(val, list):
            for m in val:
                if m:
                    mids.add(m)
        elif isinstance(val, str) and val.strip():
            # comma-separated fallback
            for m in [x.strip() for x in val.split(",") if x.strip()]:
                mids.add(m)
    # if entity has canonical_id, include canonical_members map
    cid = entity.get("canonical_id")
    if cid and canonical_members.get(cid):
        for m in canonical_members[cid]:
            mids.add(m)
    # include this entity id itself
    if entity.get("id"):
        mids.add(entity.get("id"))
    return list(mids)

# helper to gather chunk ids from member records (search in available sources)
# We'll build a small index from all loaded sources mapping id -> record
index_by_id = {}
for lst in (resolved, first_run_resolved, raw_reps):
    for r in lst:
        rid = r.get("id")
        if rid:
            index_by_id[rid] = r

# also attempt to index clustered input items
for lst in (clustered,):
    for r in lst:
        rid = r.get("id")
        if rid:
            index_by_id.setdefault(rid, r)

def collect_chunk_ids_for_members(member_ids):
    chunk_ids = set()
    for mid in member_ids:
        rec = index_by_id.get(mid)
        if not rec:
            # maybe mid is itself a chunk id (some workflows used chunk id as member id)
            # if it resembles 'Ch_' or present in chunks folder, keep it too
            if isinstance(mid, str) and mid.startswith("Ch_"):
                chunk_ids.add(mid)
            continue
        # candidate chunk id fields
        for fld in ("chunk_id","source_chunk_id","chunk_ids","ref_index","chunk_id_list"):
            val = rec.get(fld)
            if isinstance(val, list):
                chunk_ids.update([v for v in val if v])
            elif isinstance(val, str) and val.strip():
                # maybe a single chunk id or comma-separated
                if "," in val:
                    chunk_ids.update([x.strip() for x in val.split(",") if x.strip()])
                else:
                    chunk_ids.add(val)
        # some records include 'ref_index' as provenance (add it)
        if rec.get("ref_index"):
            chunk_ids.add(str(rec.get("ref_index")))
    return list(chunk_ids)

# If resolved list is empty, choose fallback source for "final" entities
primary_entities = resolved if resolved else (raw_reps if raw_reps else first_run_resolved)

if not primary_entities:
    logging.error("No primary entity source found (resolved / raw reps / first-run). Exiting.")
    raise SystemExit(1)

master_rows = []
for ent in primary_entities:
    # base copy of entity
    row = dict(ent)  # preserve all existing fields
    # ensure id present
    eid = row.get("id") or row.get("entity_id") or row.get("canonical_id") or f"Row_{len(master_rows)+1}"
    row["id"] = eid

    # collect member ids (merges over both runs)
    mids = collect_member_ids(row)
    # also check canonical_members mapping by matching row.id if it is a member of some canonical
    if row.get("id") in member_to_canonical:
        cid = member_to_canonical[row.get("id")]
        for m in canonical_members.get(cid, []):
            if m not in mids:
                mids.append(m)

    # also try to collect members from canonical record if this row has canonical_id
    if row.get("canonical_id") and canonical_members.get(row.get("canonical_id")):
        for m in canonical_members[row.get("canonical_id")]:
            if m not in mids:
                mids.append(m)

    row["member_ids_all"] = sorted(list(dict.fromkeys(mids)))  # dedupe preserve order

    # collect chunk ids for those members
    chunk_ids = collect_chunk_ids_for_members(row["member_ids_all"])
    row["chunk_ids_all"] = sorted(list(dict.fromkeys(chunk_ids)))

    # best metadata: pick canonical metadata if canonical exists
    cid = row.get("canonical_id")
    if cid:
        canon_meta = next((c for c in canons if (c.get("canonical_id") == cid or c.get("id") == cid)), None)
        if canon_meta:
            # prefer canonical name/desc/type if present
            if canon_meta.get("canonical_name"):
                row["canonical_name"] = canon_meta.get("canonical_name")
            if canon_meta.get("canonical_description"):
                row["canonical_description"] = canon_meta.get("canonical_description")
            if canon_meta.get("canonical_type"):
                row["canonical_type"] = canon_meta.get("canonical_type")

    # ensure canonical_id is present (maybe mapping exists)
    if not row.get("canonical_id"):
        # maybe this row's id maps to a canonical via member_to_canonical
        mapped_cid = member_to_canonical.get(row.get("id"))
        if mapped_cid:
            row["canonical_id"] = mapped_cid

    # ensure some canonical_name present (from canonical map or highest-confidence member)
    if not row.get("canonical_name"):
        # choose highest-confidence member's name if available
        best = None
        best_conf = -1.0
        for mid in row["member_ids_all"]:
            rec = index_by_id.get(mid)
            if rec:
                try:
                    conf = float(rec.get("confidence_score") or 0.0)
                except Exception:
                    conf = 0.0
                if conf > best_conf:
                    best_conf = conf
                    best = rec
        if best:
            row.setdefault("canonical_name", best.get("entity_name") or best.get("entity_name_original"))

    # normalize some common fields
    row["entity_name"] = row.get("entity_name") or row.get("canonical_name") or row.get("id")
    row["entity_description"] = row.get("entity_description") or row.get("canonical_description") or ""
    row["entity_type_hint"] = row.get("entity_type_hint") or row.get("canonical_type") or ""
    # confidence: prefer explicit field else try best member confidence
    if row.get("confidence_score") is None:
        best_conf = None
        for mid in row["member_ids_all"]:
            rec = index_by_id.get(mid)
            if rec:
                try:
                    c = float(rec.get("confidence_score") or 0.0)
                except Exception:
                    c = 0.0
                if best_conf is None or c > best_conf:
                    best_conf = c
        row["confidence_score"] = best_conf if best_conf is not None else None

    # chunk_text_truncated (if available as _chunk_text_truncated or similar)
    row["chunk_text_truncated"] = row.get("_chunk_text_truncated") or row.get("chunk_text_truncated") or ""

    master_rows.append(row)

logging.info(f"Built master rows: {len(master_rows)} final entities")

# ---------- Write master JSONL + JSON ----------
MASTER_JSONL = FINAL_DIR / "final_entities_master.jsonl"
MASTER_JSON = FINAL_DIR / "final_entities_master.json"

write_jsonl(MASTER_JSONL, master_rows)
write_json(MASTER_JSON, master_rows)

# ---------- Produce simple summaries & plots ----------
# 1) canonical_summary.csv (one row per canonical group present in master)
canon_summary_rows = []
for cid, members in canonical_members.items():
    # find canonical metadata if available
    cmeta = next((c for c in canons if c.get("canonical_id") == cid or c.get("id") == cid), {})
    canon_summary_rows.append({
        "canonical_id": cid,
        "canonical_name": cmeta.get("canonical_name") or "",
        "canonical_type": cmeta.get("canonical_type") or "",
        "n_members": len(members),
        "member_sample": " | ".join(members[:10]),
        "rationale": cmeta.get("rationale") or ""
    })

CANON_SUM_CSV = FINAL_DIR / "canonical_summary.csv"
with open(CANON_SUM_CSV, "w", newline="", encoding="utf-8") as fh:
    fieldnames = ["canonical_id","canonical_name","canonical_type","n_members","member_sample","rationale"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in canon_summary_rows:
        writer.writerow(r)
logging.info(f"Wrote canonical summary CSV: {CANON_SUM_CSV}")

# 2) merged_groups.json (copy canonical_members mapping)
MERGED_GROUPS_JSON = FINAL_DIR / "merged_groups_by_canonical.json"
write_json(MERGED_GROUPS_JSON, canonical_members)

# 3) actions summary from master (resolved_action distribution)
action_counts = Counter()
for r in master_rows:
    action_counts.update([r.get("resolved_action") or "<none>"])

ACTIONS_JSON = FINAL_DIR / "actions_summary.json"
write_json(ACTIONS_JSON, dict(action_counts))

# 4) produce plots if matplotlib available
if plt is not None and pd is not None:
    try:
        # merges histogram
        sizes = [len(v) for v in canonical_members.values()] if canonical_members else []
        if sizes:
            plt.figure(figsize=(6,4))
            plt.hist(sizes, bins=range(1, max(sizes)+2), edgecolor="black", align="left")
            plt.title("Distribution of canonical group sizes (# members)")
            plt.xlabel("members per canonical entity")
            plt.ylabel("count")
            plt.tight_layout()
            merges_hist = FINAL_DIR / "merges_hist.png"
            plt.savefig(merges_hist, dpi=150)
            plt.close()
            logging.info(f"Wrote plot: {merges_hist}")

        # actions pie (top 8)
        if action_counts:
            actions_df = pd.DataFrame(action_counts.items(), columns=["action","count"]).sort_values("count", ascending=False)
            top_actions = actions_df.head(8)
            plt.figure(figsize=(6,6))
            plt.pie(top_actions["count"], labels=top_actions["action"], autopct="%1.1f%%", startangle=140)
            plt.title("Top resolved_action distribution")
            plt.tight_layout()
            actions_pie = FINAL_DIR / "actions_pie.png"
            plt.savefig(actions_pie, dpi=150)
            plt.close()
            logging.info(f"Wrote plot: {actions_pie}")
    except Exception as e:
        logging.warning(f"Plot generation failed: {e}")
else:
    logging.warning("pandas/matplotlib not available; skipping plot generation. Install them with pip to enable plots.")

# ---------- Copy over any existing analysis files into FinalCompAnalysis ----------
# Copy common filenames / filetypes from Entities tree that look useful (csv,json,png,md)
copied = []
for pattern in ("**/*entResAnalysis*.json", "**/*entResAnalysis*.csv", "**/*entResAnalysis*.png",
                "**/*analysis*.json", "**/*analysis*.csv", "**/*Analysis*.png", "**/*clusters_summary*.json",
                "**/*canonical_entities*.jsonl", "**/*canonical_entities*.json", "**/*entities_resolved*.jsonl",
                "**/*entities_clustered*.jsonl", "**/*merged_groups*.json", "**/*top50_canonical*.csv"):
    for p in SEARCH_ROOT.glob(pattern):
        # avoid copying master outputs we just created
        if FINAL_DIR in p.parents:
            continue
        dst = FINAL_DIR / p.name
        try:
            shutil.copy2(p, dst)
            copied.append(p)
        except Exception as e:
            logging.warning(f"Failed to copy {p}: {e}")

logging.info(f"Copied {len(copied)} existing analysis files into output folder (if any)")

# ---------- Create a small README and summary markdown ----------
README = FINAL_DIR / "README.md"
report_md = FINAL_DIR / "FinalCompAnalysis_Summary.md"

readme_text = f"""# FinalCompAnalysis

Folder generated by `create_final_comp_analysis.py`

Location: `{FINAL_DIR}`

Contents:
- `final_entities_master.jsonl` : master entity file (one JSON object per final entity after the second run).
- `final_entities_master.json`  : same as above but stored as a JSON array.
- `canonical_summary.csv`      : summary per canonical group (member counts, sample).
- `merged_groups_by_canonical.json` : mapping canonical_id -> member ids aggregated from available records.
- `actions_summary.json`       : distribution of resolution actions (counts).
- `merges_hist.png`, `actions_pie.png` : diagnostic plots (if generated).
- Copied analysis files from repo (if found).

How to use:
1. Use `final_entities_master.jsonl` as the input for your next step (class identification).
2. Use `canonical_summary.csv` and `merged_groups_by_canonical.json` for reporting in your paper (ACL).
"""

write_json(README, {"note": "See FinalCompAnalysis_Summary.md for human readable summary"})  # also write JSON for machine-readability
with open(README, "w", encoding="utf-8") as fh:
    fh.write(readme_text)
logging.info(f"Wrote README: {README}")

# human-readable report
report_lines = []
report_lines.append("# FinalCompAnalysis Summary\n")
report_lines.append(f"**Output folder:** `{FINAL_DIR}`\n")
report_lines.append(f"- Master rows (final entities): **{len(master_rows)}**\n")
report_lines.append(f"- Canonical groups aggregated: **{len(canonical_members)}**\n")
report_lines.append(f"- Action counts: {dict(action_counts)}\n")
report_lines.append("\n## Input files used (first matches from candidate lists)\n")
report_lines.append(f"- resolved (primary): `{resolved_path}`\n")
report_lines.append(f"- canonical records: `{canon_path}`\n")
report_lines.append(f"- clustered input (if any): `{clustered_path}`\n")
report_lines.append(f"- raw/reps used (if any): `{raw_rep_path}`\n")
report_lines.append("\n## Files produced\n")
for f in sorted(FINAL_DIR.iterdir()):
    report_lines.append(f"- `{f.name}`")
report_md_text = "\n".join(report_lines)
with open(report_md, "w", encoding="utf-8") as fh:
    fh.write(report_md_text)
logging.info(f"Wrote summary report: {report_md}")

# ---------- Final log output to console ----------
logging.info("FinalCompAnalysis build complete.")
logging.info(f"See folder: {FINAL_DIR}")
logging.info("If something is missing, run the script again after making sure the repo files exist at expected locations.")

# Exit





#endregion#? Final Analysis
#?#########################  End  ##########################






#endregion#! Old Ent Res Manual Re Run
#!#############################################  End Chapter  ##################################################























#endregion#! Entity Identification
#!#############################################  End Chapter  ##################################################







#!############################################# Start Chapter ##################################################
#region:#!   Class Identification




#*######################### Start ##########################
#region:#?   Cls Rec V1

#!/usr/bin/env python3
"""
classrec_iterative.py

Iterative Class Recognition (ClassRec) aligned with your pipeline.
- Input: resolved entities JSONL (one entity object per line) at:
    /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl
- Output directory:
    /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec
Produces:
 - class_candidates.jsonl   (LLM-suggested classes, one JSON object per line)
 - cluster_to_members.json  (coarse cluster summary used for first pass)
 - remaining_entities.jsonl (entities left after process)
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client (same pattern as your other scripts)
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CANDIDATES_OUT = OUT_DIR / "class_candidates.jsonl"
CLUSTER_SUMMARY_OUT = OUT_DIR / "cluster_to_members.json"
REMAINING_ENTITIES_OUT = OUT_DIR / "remaining_entities.jsonl"

# embedder / model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}

# HDBSCAN + UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0

# local subcluster
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# prompt and LLM / limits
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 2500

# iteration control
MAX_RECLUSTER_ROUNDS = 8  # safety cap to avoid infinite loops
VERBOSE = True

# ------------------------ OpenAI client loader -----------------------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ------------------------- HF Embedder ------------------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ----------------------------------
def load_entities(path: Path) -> List[Dict]:
    assert path.exists(), f"Input not found: {path}"
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ents.append(json.loads(line))
    return ents

def safe_text(e: Dict, k: str) -> str:
    v = e.get(k)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or "")
        descs.append(safe_text(e, "entity_description") or "")
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or ""
        et = safe_text(e, "entity_type_hint") or ""
        # node props to text
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
            if pieces:
                node_props_text = " | ".join(pieces)
        parts = []
        if et:
            parts.append(f"[TYPE:{et}]")
        if resolution:
            parts.append(resolution)
        if node_props_text:
            parts.append(node_props_text)
        ctxs.append(" ; ".join(parts))
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name); emb_desc = _ensure(emb_desc); emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0); w_desc = weights.get("desc", 0.0); w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0: raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0]-1)),
                            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0]-1)),
                            min_dist=UMAP_MIN_DIST,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ------------------ Prompt (revised) --------------------------------
CLASS_PROMPT_TEMPLATE = """
You are a careful schema/ontology suggester.

Goal (short): given a small list of entity mentions (name + short description + evidence excerpt),
suggest zero or more *class* concepts that summarize natural groups among these members.

Important rule about WHEN TO CREATE A CLASS:
- Prefer classes that have at least TWO members that naturally belong together.
- **Only** create a single-member class when your input contains exactly ONE entity (i.e., you were given only one member).
- If you are given multiple members but you think one member could be a standalone class, do NOT create a class for it now; leave it unassigned. Single-member classes are allowed only in the single-entity prompt case.

How we define 'useful' classes (guide for granularity):
- Not too broad (avoid "everything" or overly general labels).
- Not too narrow (avoid trivial singletons when a meaningful group exists).
- A useful class should allow different entities to connect to the same schema concept (practical reuse).

Return ONLY a JSON ARRAY. Each element must be an object with keys:
 - class_label (string): short canonical class name (1-3 words)
 - class_description (string): 1-2 sentence description explaining the class
 - member_ids (array[string]): entity ids (must be members from the provided list)
 - confidence (float): 0.0-1.0 estimate of confidence
 - evidence_excerpt (string): 5-30 word excerpt that supports why these members form this class (optional)

Rules:
- Use ONLY the provided members (do not invent other ids or outside facts).
- Aim for non-overlapping classes when possible (overlap allowed if really justified).
- If you cannot propose any sensible class, return an empty array [].
- Keep labels short and meaningful; use description to give nuance.

Members (one per line: id | name | description | evidence_excerpt):
{members_block}

Return JSON array only.
"""

def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id")
        name = (m.get("entity_name") or "")[:120].replace("\n"," ")
        desc = (m.get("entity_description") or "")[:300].replace("\n"," ")
        evidence = (m.get("resolution_context") or m.get("context_phrase") or "")[:300].replace("\n"," ")
        rows.append(f"{eid} | {name} | {desc} | {evidence}")
    return "\n".join(rows)

def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    s = txt.strip()
    # remove fences
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(members: List[Dict], single_entity_mode: bool = False) -> List[Dict]:
    """
    members: list of entity dicts (these must contain id/name/desc/resolution_context)
    single_entity_mode: if True the prompt will be for a single entity (allowed to propose a single-member class)
    Returns: list of class candidate dicts produced by LLM (may be empty)
    """
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE: print(f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}")
        return []
    llm_out = call_llm(prompt)
    arr = parse_json_array_from_text(llm_out)
    if not arr:
        return []
    candidates = []
    for c in arr:
        label = c.get("class_label") or c.get("label") or c.get("name")
        if not label:
            continue
        member_ids = c.get("member_ids") or c.get("members") or []
        confidence = float(c.get("confidence")) if c.get("confidence") is not None else 0.0
        desc = c.get("class_description") or c.get("description") or ""
        ev = c.get("evidence_excerpt") or ""
        candidate = {
            "candidate_id": "ClsC_" + uuid.uuid4().hex[:8],
            "class_label": label,
            "class_description": desc,
            "member_ids": member_ids,
            "confidence": confidence,
            "evidence_excerpt": ev,
            "created_from_ids": [m["id"] for m in members],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        candidates.append(candidate)
    return candidates

# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main():
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    # ensure ids
    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    # prepare embedder and embeddings
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    # initial coarse clustering
    labels, _ = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    # cluster membership mapping
    cluster_to_indices = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    # bookkeeping sets
    seen_by_llm = set()               # seen by LLM (appeared in any prompt)
    assigned_entity_ids = set()       # entities assigned to some class
    class_candidates = []             # accumulated candidates

    # process non-noise clusters first (lab != -1)
    def process_cluster_indices(indices: List[int]):
        nonlocal class_candidates, seen_by_llm, assigned_entity_ids
        # if small cluster -> chunk directly
        if len(indices) <= MAX_CLUSTER_SIZE_FOR_LOCAL:
            # chunk into MAX_MEMBERS_PER_PROMPT
            for i in range(0, len(indices), MAX_MEMBERS_PER_PROMPT):
                chunk_idxs = indices[i:i+MAX_MEMBERS_PER_PROMPT]
                members = [entities[j] for j in chunk_idxs]
                # call LLM and collect classes
                candidates = process_member_chunk_llm(members, single_entity_mode=(len(members)==1))
                # record seen ids
                for m in members:
                    seen_by_llm.add(m["id"])
                # record produced classes and assigned members
                for c in candidates:
                    # keep only classes that reference at least two member_ids if chunk size>1, otherwise allow 1-member if chunk was single
                    mids = c.get("member_ids", [])
                    # Filter out any member ids not in provided members (safety)
                    mids = [mid for mid in mids if mid in [m["id"] for m in members]]
                    if not mids:
                        continue
                    if len(members) == 1:
                        # single-entity prompt: allow 1-member classes
                        pass
                    else:
                        # multi-entity prompt: enforce >=2 members
                        if len(mids) < 2:
                            # LLM proposed a 1-member class while given many members -> ignore (as instructed)
                            continue
                    # accept class
                    c["member_ids"] = mids
                    class_candidates.append(c)
                    for mid in mids:
                        assigned_entity_ids.add(mid)
        else:
            # large cluster: local HDBSCAN subcluster
            sub_emb = combined_emb[indices]
            try:
                local_lab = hdbscan.HDBSCAN(min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            metric="euclidean", cluster_selection_method='eom')
                local_labels = local_lab.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(indices), dtype=int)
            # group by local label and process non-noise local groups
            local_groups = {}
            for i_local, lab_local in enumerate(local_labels):
                local_groups.setdefault(int(lab_local), []).append(indices[i_local])
            for llab, idxs in local_groups.items():
                if llab == -1:
                    # skip local noise here; will be processed in later iterative passes
                    # but still mark them as seen by being included? NO: they were not shown to LLM in local grouping
                    # we'll not mark them as seen here
                    continue
                # process each subgroup (recursive)
                process_cluster_indices(idxs)

    # initial pass: process all non-noise coarse clusters
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE: print(f"[pass0] processing coarse cluster {lab} size={len(idxs)}")
        process_cluster_indices(idxs)

    # iterative recluster loop:
    # remaining_entities = (original - assigned) and also include those that were seen_by_llm but not assigned
    all_entity_ids = [e["id"] for e in entities]
    # initial set of unassigned includes cluster -1 plus seen-but-unassigned
    initial_noise_indices = cluster_to_indices.get(-1, [])
    # track indices (not ids) to recluster
    def ids_to_indices(id_list):
        id_to_idx = {e["id"]: i for i, e in enumerate(entities)}
        return [id_to_idx[iid] for iid in id_list if iid in id_to_idx]

    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        # construct current pool indices: (original noise indices) + (seen-by-llm but not assigned)
        seen_but_unassigned_ids = list(seen_by_llm - assigned_entity_ids)
        pool_ids = set([entities[i]["id"] for i in initial_noise_indices] + seen_but_unassigned_ids)
        # remove already-assigned just-in-case
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE: print(f"[iter {round_num}] pool empty -> stopping recluster loop")
            break
        pool_indices = ids_to_indices(pool_ids)
        if not pool_indices:
            if VERBOSE: print(f"[iter {round_num}] pool indices empty -> stopping")
            break

        if VERBOSE: print(f"[iter {round_num}] reclustering pool size={len(pool_indices)} (noise + seen-but-unassigned)")

        # build sub-emb and cluster them
        sub_emb = combined_emb[pool_indices]
        try:
            labels_sub, _ = run_hdbscan(sub_emb, min_cluster_size=2, min_samples=1, use_umap=False)
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        # map label -> indices (global)
        sub_cluster_map = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        new_classes_in_round = 0
        # process non-noise subclusters
        for lab_sub, gidxs in sorted(sub_cluster_map.items(), key=lambda x: (x[0]==-1, x[0])):
            if lab_sub == -1:
                continue
            if VERBOSE: print(f"[iter {round_num}] processing subcluster {lab_sub} size={len(gidxs)}")
            # process cluster indices normally (this will call LLM)
            before_assigned = len(assigned_entity_ids)
            process_cluster_indices(gidxs)
            after_assigned = len(assigned_entity_ids)
            new_classes_in_round += (after_assigned - before_assigned)

        if new_classes_in_round == 0:
            if VERBOSE: print(f"[iter {round_num}] no new assignments produced this round -> stopping recluster loop")
            break
        else:
            if VERBOSE: print(f"[iter {round_num}] new assignments this round: {new_classes_in_round}")
            # continue to next round to try to resolve more

    # After iterative reclustering, prepare final remaining entity list:
    assigned_ids = set(assigned_entity_ids)
    remaining_entities = [e for e in entities if e["id"] not in assigned_ids]

    # FINAL single-entity pass: allow single-member class proposals
    if VERBOSE: print(f"[final] single-entity pass on {len(remaining_entities)} remaining entities")
    for e in remaining_entities:
        # single-entity prompt
        candidates = process_member_chunk_llm([e], single_entity_mode=True)
        # accept classes produced for this single-entity prompt
        for c in candidates:
            # ensure member_ids valid: if LLM returned different id, normalize to the entity id we provided
            if not c.get("member_ids"):
                c["member_ids"] = [e["id"]]
            # accept
            class_candidates.append(c)
            assigned_entity_ids.update(c["member_ids"])

    # recompute remaining after single-entity pass
    assigned_ids = set(assigned_entity_ids)
    remaining_entities = [e for e in entities if e["id"] not in assigned_ids]

    # Save outputs
    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in class_candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    # coarse cluster summary output for inspection
    cluster_summary = {int(k): [entities[i]["id"] for i in v] for k, v in sorted(cluster_to_indices.items(), key=lambda x: x[0])}
    with open(CLUSTER_SUMMARY_OUT, "w", encoding="utf-8") as fh:
        json.dump(cluster_summary, fh, ensure_ascii=False, indent=2)

    with open(REMAINING_ENTITIES_OUT, "w", encoding="utf-8") as fh:
        for e in remaining_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(class_candidates)} class candidates -> {CLASS_CANDIDATES_OUT}")
    print(f"[done] wrote coarse cluster summary -> {CLUSTER_SUMMARY_OUT}")
    print(f"[done] wrote {len(remaining_entities)} remaining entities -> {REMAINING_ENTITIES_OUT}")


if __name__ == "__main__":
    classrec_iterative_main()


#endregion#? Cls Rec V1
#*#########################  End  ##########################




#*######################### Start ##########################
#region:#?   Cls Rec V2

#!/usr/bin/env python3
"""
classrec_iterative.py  (REVISED)

Iterative Class Recognition (ClassRec) aligned with your pipeline.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl

Outputs (directory):
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/
    - class_candidates.jsonl          (LLM-suggested classes; each line = JSON object)
    - initial_cluster_entities.json   (coarse initial clusters; full entity objects included)
    - recluster_round_{i}.json        (recluster outputs for each iterative round)
    - final_unassigned.jsonl          (full entity objects that remain unassigned after single-entity pass)
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client (same pattern)
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CANDIDATES_OUT = OUT_DIR / "class_candidates.jsonl"
INITIAL_CLUSTER_OUT = OUT_DIR / "initial_cluster_entities.json"
RECLUSTER_PREFIX = OUT_DIR / "recluster_round_"
FINAL_UNASSIGNED_OUT = OUT_DIR / "final_unassigned.jsonl"

# embedder / model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}

# HDBSCAN + UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0

# local subcluster
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# prompt and LLM / limits
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 2500

# iteration control
MAX_RECLUSTER_ROUNDS = 8  # safety cap
VERBOSE = True

# ------------------------ OpenAI client loader -----------------------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ------------------------- HF Embedder ------------------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ----------------------------------
def load_entities(path: Path) -> List[Dict]:
    assert path.exists(), f"Input not found: {path}"
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ents.append(json.loads(line))
    return ents

def safe_text(e: Dict, k: str) -> str:
    v = e.get(k)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or "")
        descs.append(safe_text(e, "entity_description") or "")
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or ""
        et = safe_text(e, "entity_type_hint") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
            if pieces:
                node_props_text = " | ".join(pieces)
        parts = []
        if et:
            parts.append(f"[TYPE:{et}]")
        if resolution:
            parts.append(resolution)
        if node_props_text:
            parts.append(node_props_text)
        ctxs.append(" ; ".join(parts))
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name); emb_desc = _ensure(emb_desc); emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0); w_desc = weights.get("desc", 0.0); w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0: raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0]-1)),
                            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0]-1)),
                            min_dist=UMAP_MIN_DIST,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ------------------ Prompt (REVISED to be MUST) -----------------------
CLASS_PROMPT_TEMPLATE = """
You are a careful schema/ontology suggester.

Goal (short): given a small list of entity mentions (name + short description + evidence excerpt),
suggest zero or more *class* concepts that summarize natural groups among these members.

Important rule about WHEN TO CREATE A CLASS (ENFORCED):
- When the prompt contains MULTIPLE members you MUST create only classes that contain TWO OR MORE members.
- You MUST NOT create a single-member class in a prompt that contains multiple members.
- The ONLY situation in which a single-member class is allowed is when the input prompt contains EXACTLY ONE entity (single-entity mode).

How we define 'useful' classes (guide for granularity):
- Not too broad (avoid "everything" or overly general labels).
- Not too narrow (avoid trivial singletons when a meaningful group exists).
- A useful class should allow different entities to connect to the same schema concept (practical reuse).

Return ONLY a JSON ARRAY. Each element must be an object with keys:
 - class_label (string): short canonical class name (1-3 words)
 - class_description (string): 1-2 sentence description explaining the class
 - member_ids (array[string]): entity ids (must be members from the provided list)
 - confidence (float): 0.0-1.0 estimate of confidence
 - evidence_excerpt (string): 5-30 word excerpt that supports why these members form this class (optional)

Rules:
- Use ONLY the provided members (do not invent other ids or outside facts).
- Aim for non-overlapping classes when possible (overlap allowed if really justified).
- If you cannot propose any sensible class, return an empty array [].
- Keep labels short and meaningful; use description to give nuance.

Members (one per line: id | name | description | evidence_excerpt):
{members_block}

Return JSON array only.
"""

def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id")
        name = (m.get("entity_name") or "")[:120].replace("\n"," ")
        desc = (m.get("entity_description") or "")[:300].replace("\n"," ")
        evidence = (m.get("resolution_context") or m.get("context_phrase") or "")[:300].replace("\n"," ")
        rows.append(f"{eid} | {name} | {desc} | {evidence}")
    return "\n".join(rows)

def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    s = txt.strip()
    # remove fences
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(members: List[Dict], single_entity_mode: bool = False) -> List[Dict]:
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE: print(f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}")
        return []
    llm_out = call_llm(prompt)
    arr = parse_json_array_from_text(llm_out)
    if not arr:
        return []
    candidates = []
    provided_ids = {m["id"] for m in members}
    for c in arr:
        label = c.get("class_label") or c.get("label") or c.get("name")
        if not label:
            continue
        member_ids = c.get("member_ids") or c.get("members") or []
        # sanitize: keep only member ids that were in provided list
        member_ids = [mid for mid in member_ids if mid in provided_ids]
        if not member_ids:
            continue
        # Enforce rule: if multi-member prompt (len(members)>1), require >=2 member_ids
        if not single_entity_mode and len(members) > 1 and len(member_ids) < 2:
            # reject this candidate as it violates the MUST rule
            continue
        confidence = float(c.get("confidence")) if c.get("confidence") is not None else 0.0
        desc = c.get("class_description") or c.get("description") or ""
        ev = c.get("evidence_excerpt") or ""
        candidate = {
            "candidate_id": "ClsC_" + uuid.uuid4().hex[:8],
            "class_label": label,
            "class_description": desc,
            "member_ids": member_ids,
            "confidence": confidence,
            "evidence_excerpt": ev,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        candidates.append(candidate)
    return candidates

# -------------------- Utility: write cluster files (full entity objects) -----------
def write_cluster_summary(path: Path, cluster_map: Dict[int, List[int]], entities: List[Dict]):
    """
    Writes a JSON file with:
    {
      "n_entities": N,
      "n_clusters": K,
      "clusters": {
         "0": [ <entityobj compact line>, <entityobj compact line>, ... ],
         "1": [ ... ]
      }
    }
    Each entity object is the full entity dict from input (compact single-line JSON for each entity)
    """
    n_entities = len(entities)
    clusters = {}
    for k, idxs in sorted(cluster_map.items(), key=lambda x: x[0]):
        arr = []
        for i in idxs:
            ent = entities[i]
            arr.append(ent)  # full object
        clusters[str(k)] = arr
    meta = {"n_entities": n_entities, "n_clusters": len(clusters), "clusters": clusters}
    # write top-level pretty but ensure entity objects are compact in arrays
    # We'll dump manually to control entity formatting: pretty top-level, but array elements compact single-line
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')
        cluster_items = list(clusters.items())
        for ci, (k, ents) in enumerate(cluster_items):
            fh.write(f'    "{k}": [\n')
            for ei, ent in enumerate(ents):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'      {ent_json}')
                if ei < len(ents) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('    ]')
            if ci < len(cluster_items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main():
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    # ensure ids exist
    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    # prepare embedder and embeddings
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    # initial coarse clustering
    labels, _ = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    # cluster membership mapping
    cluster_to_indices = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    # save initial clustering with full entity objects
    write_cluster_summary(INITIAL_CLUSTER_OUT, cluster_to_indices, entities)
    if VERBOSE: print(f"[write] initial cluster file -> {INITIAL_CLUSTER_OUT}")

    # bookkeeping sets
    seen_by_llm = set()
    assigned_entity_ids = set()
    class_candidates = []

    # helper to map id->index
    id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    # worker to process a list of global indices
    def process_cluster_indices(indices: List[int]):
        nonlocal class_candidates, seen_by_llm, assigned_entity_ids
        if len(indices) == 0:
            return
        # chunk handling for moderate clusters
        if len(indices) <= MAX_CLUSTER_SIZE_FOR_LOCAL:
            for i in range(0, len(indices), MAX_MEMBERS_PER_PROMPT):
                chunk_idxs = indices[i:i+MAX_MEMBERS_PER_PROMPT]
                members = [entities[j] for j in chunk_idxs]
                # call LLM
                candidates = process_member_chunk_llm(members, single_entity_mode=(len(members) == 1))
                # mark seen
                for m in members:
                    seen_by_llm.add(m["id"])
                # accept candidates and attach member_entities
                for c in candidates:
                    mids = c.get("member_ids", [])
                    # attach full member entity objects
                    member_entities = [entities[id_to_index[mid]] for mid in mids if mid in id_to_index]
                    if not member_entities:
                        continue
                    c["member_entities"] = member_entities
                    # append candidate and mark assigned entities
                    class_candidates.append(c)
                    for mid in mids:
                        assigned_entity_ids.add(mid)
        else:
            # large cluster -> local subcluster via HDBSCAN
            sub_emb = combined_emb[indices]
            try:
                local_lab = hdbscan.HDBSCAN(min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            metric="euclidean", cluster_selection_method='eom')
                local_labels = local_lab.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(indices), dtype=int)
            local_groups = {}
            for i_local, lab_local in enumerate(local_labels):
                local_groups.setdefault(int(lab_local), []).append(indices[i_local])
            for llab, idxs in local_groups.items():
                if llab == -1:
                    continue
                process_cluster_indices(idxs)

    # initial pass: all non-noise clusters
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE: print(f"[pass0] processing coarse cluster {lab} size={len(idxs)}")
        process_cluster_indices(idxs)

    # iterative recluster loop: combine original noise (-1) + seen-but-unassigned
    initial_noise_indices = cluster_to_indices.get(-1, [])
    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        seen_but_unassigned_ids = list(seen_by_llm - assigned_entity_ids)
        pool_ids = set([entities[i]["id"] for i in initial_noise_indices] + seen_but_unassigned_ids)
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE: print(f"[iter {round_num}] pool empty -> stopping recluster loop")
            break
        pool_indices = [id_to_index[pid] for pid in pool_ids if pid in id_to_index]
        if not pool_indices:
            if VERBOSE: print(f"[iter {round_num}] pool indices empty -> stopping")
            break

        if VERBOSE: print(f"[iter {round_num}] reclustering pool size={len(pool_indices)}")
        sub_emb = combined_emb[pool_indices]
        try:
            labels_sub, _ = run_hdbscan(sub_emb, min_cluster_size=2, min_samples=1, use_umap=False)
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        # map label -> global indices
        sub_cluster_map = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        # save this recluster round summary (full entities)
        recluster_path = Path(f"{RECLUSTER_PREFIX}{round_num}.json")
        write_cluster_summary(recluster_path, sub_cluster_map, entities)
        if VERBOSE: print(f"[write] recluster round {round_num} -> {recluster_path}")

        new_assignments = 0
        # process non-noise subclusters
        for lab_sub, gidxs in sorted(sub_cluster_map.items(), key=lambda x: (x[0]==-1, x[0])):
            if lab_sub == -1:
                continue
            before = len(assigned_entity_ids)
            if VERBOSE: print(f"[iter {round_num}] processing subcluster {lab_sub} size={len(gidxs)}")
            process_cluster_indices(gidxs)
            after = len(assigned_entity_ids)
            new_assignments += (after - before)

        if new_assignments == 0:
            if VERBOSE: print(f"[iter {round_num}] no new assignments -> stopping recluster loop")
            break
        else:
            if VERBOSE: print(f"[iter {round_num}] new assignments this round: {new_assignments}")
            # continue loop to try to resolve more

    # Final single-entity pass: allow single-member class proposals for remaining entities
    assigned_ids = set(assigned_entity_ids)
    remaining_entities = [e for e in entities if e["id"] not in assigned_ids]
    if VERBOSE: print(f"[final] single-entity pass on {len(remaining_entities)} remaining entities")
    for e in remaining_entities:
        candidates = process_member_chunk_llm([e], single_entity_mode=True)
        for c in candidates:
            # ensure member_ids valid (fall back to provided id)
            if not c.get("member_ids"):
                c["member_ids"] = [e["id"]]
            # attach full member_entities
            mids = c["member_ids"]
            member_entities = [entities[id_to_index[mid]] for mid in mids if mid in id_to_index]
            c["member_entities"] = member_entities or [e]
            class_candidates.append(c)
            for mid in c["member_ids"]:
                assigned_entity_ids.add(mid)

    # recompute final remaining after single-entity pass
    assigned_ids = set(assigned_entity_ids)
    final_remaining = [e for e in entities if e["id"] not in assigned_ids]

    # Save class candidates (each line = JSON)
    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in class_candidates:
            # remove any internal-only fields if present (none should be)
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[write] class candidates -> {CLASS_CANDIDATES_OUT}  (count={len(class_candidates)})")

    # Save final remaining unassigned full entities
    with open(FINAL_UNASSIGNED_OUT, "w", encoding="utf-8") as fh:
        for e in final_remaining:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[write] final unassigned entities -> {FINAL_UNASSIGNED_OUT}  (count={len(final_remaining)})")

    print("[done] ClassRec iterative finished.")

if __name__ == "__main__":
    classrec_iterative_main()



#endregion#? Cls Rec V2
#*#########################  End  ##########################






#*######################### Start ##########################
#region:#?   Cls Rec V3


#!/usr/bin/env python3


"""
classrec_iterative_v3.py

Iterative Class Recognition (ClassRec) with per-iteration class outputs
matching the cluster-file visual/JSON format and including class metadata
(label/desc/confidence/evidence + source cluster id).
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CANDIDATES_OUT = OUT_DIR / "class_candidates.jsonl"
INITIAL_CLUSTER_OUT = OUT_DIR / "initial_cluster_entities.json"
RECLUSTER_PREFIX = OUT_DIR / "recluster_round_"
CLASSES_PREFIX = OUT_DIR / "classes_round_"
REMAINING_OUT = OUT_DIR / "remaining_entities.jsonl"

# embedder / model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}

# HDBSCAN + UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0

# local subcluster
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# prompt and LLM / limits
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 2500

# iteration control
MAX_RECLUSTER_ROUNDS = 12  # safety cap
VERBOSE = True

# ------------------------ OpenAI client loader -----------------------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ------------------------- HF Embedder ------------------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ----------------------------------
def load_entities(path: Path) -> List[Dict]:
    assert path.exists(), f"Input not found: {path}"
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ents.append(json.loads(line))
    return ents

def safe_text(e: Dict, k: str) -> str:
    v = e.get(k)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name"        ) or "")
        descs.append(safe_text(e, "entity_description" ) or "")
        resolution = safe_text(e, "resolution_context" ) or safe_text(e, "text_span") or safe_text(e, "context_phrase") or ""
        et = safe_text(e,         "entity_type_hint"   ) or ""
        node_props = e.get(       "node_properties"    ) or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
            if pieces:
                node_props_text = " | ".join(pieces)
        parts = []
        if et:
            parts.append(f"[TYPE:{et}]")
        if resolution:
            parts.append(resolution)
        if node_props_text:
            parts.append(node_props_text)
        ctxs.append(" ; ".join(parts))
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name); emb_desc = _ensure(emb_desc); emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0); w_desc = weights.get("desc", 0.0); w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0: raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0]-1)),
                            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0]-1)),
                            min_dist=UMAP_MIN_DIST,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ------------------ Prompt (REVISED to be MUST) -----------------------
CLASS_PROMPT_TEMPLATE = """
You are a careful schema / ontology suggester. Be conservative and precise.

Goal (short)
Given a small list of entity mentions (name + short description + evidence excerpt), suggest zero or more *class* concepts that summarize *natural, practically useful* groups among these members.

Hard rule about WHEN TO CREATE A CLASS (ENFORCED)
- If the input contains MULTIPLE members, you MUST only propose classes that contain **TWO OR MORE** members.
- You MUST NOT invent or return a single-member class when the input contains multiple members.
- The ONLY time a single-member class is allowed is when the input prompt contains **EXACTLY ONE** entity (single-entity mode).

Primary principle (do not force-fit)
- **Do not** change or broaden a proposed class label just to make room for extra entities. If an entity does not genuinely belong, omit it.
- It is preferable to **not** propose a class at all than to propose a loose/broad class that subsumes heterogeneous entities.
- Entities omitted here are *not* lost — they will be revisited in later reclustering/single-entity passes and may find better classmates. Do not "burn" entities by forcing them into a poor class.

What makes a *useful* class (guidance on granularity)
- Not so broad that it includes everything (avoid generic catch-alls).
- Not so narrow that it becomes an unusable singleton (except single-entity mode).
- Should enable practical reuse: different entities of this class could be connected to the same schema concept and benefit downstream tasks.

Label style & description guidance
- `class_label`: short, noun-phrase (1–3 words), Title Case (e.g., "Bearing Wear", "Cooling System").
- `class_description`: 1–2 sentences describing the defining properties and boundaries of the class (what is in / what is out).
- `evidence_excerpt`: a short 5–30 word excerpt explaining why these members belong together (can be drawn from the provided evidence).

Output format (REQUIRED)
Return ONLY a JSON ARRAY. Each element must be an object with these keys:
 - class_label (string) — short canonical class name (1–3 words)
 - class_description (string) — 1–2 sentence explanation of the class
 - member_ids (array[string]) — entity ids (must be from the provided list)
 - confidence (float) — 0.0–1.0 estimate of confidence
 - evidence_excerpt (string, optional) — short supporting excerpt

Hard rules
- Use ONLY the provided members (do NOT invent ids or fetch outside facts).
- Member ids in `member_ids` must come from the input list; discard any external ids.
- If you cannot propose any sensible class, return an empty array `[]`.
- Prefer **non-overlapping** classes when possible; small, justified overlap is allowed but explain via the description/evidence.
- If a candidate class would require renaming into a much broader label to include marginal members, **reject** that merge — instead omit the marginal members.

Operational notes for scoring & downstream use
- Give honest confidence scores: higher when members share explicit lexical/semantic evidence; lower for looser semantic groupings.
- Keep labels short and stable; avoid ad-hoc punctuation or excessively long names.
- Provide concise evidence to help later canonicalization and merging.

Members (one per line: id | name | description | evidence_excerpt):
{members_block}

Return JSON array only.
"""


def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id")
        name = (m.get("entity_name") or "")[:120].replace("\n"," ")
        desc = (m.get("entity_description") or "")[:300].replace("\n"," ")
        evidence = (m.get("resolution_context") or m.get("context_phrase") or "")[:300].replace("\n"," ")
        rows.append(f"{eid} | {name} | {desc} | {evidence}")
    return "\n".join(rows)

def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    s = txt.strip()
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(members: List[Dict], single_entity_mode: bool = False) -> List[Dict]:
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE: print(f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}")
        return []
    llm_out = call_llm(prompt)
    arr = parse_json_array_from_text(llm_out)
    if not arr:
        return []
    candidates = []
    provided_ids = {m["id"] for m in members}
    for c in arr:
        label = c.get("class_label") or c.get("label") or c.get("name")
        if not label:
            continue
        member_ids = c.get("member_ids") or c.get("members") or []
        # sanitize: keep only member ids that were in provided list
        member_ids = [mid for mid in member_ids if mid in provided_ids]
        if not member_ids:
            continue
        # Enforce rule: if multi-member prompt (len(members)>1), require >=2 member_ids
        if not single_entity_mode and len(members) > 1 and len(member_ids) < 2:
            # reject this candidate as it violates the MUST rule
            continue
        confidence = float(c.get("confidence")) if c.get("confidence") is not None else 0.0
        desc = c.get("class_description") or c.get("description") or ""
        ev = c.get("evidence_excerpt") or ""
        candidate = {
            "candidate_id": "ClsC_" + uuid.uuid4().hex[:8],
            "class_label": label,
            "class_description": desc,
            "member_ids": member_ids,
            "confidence": confidence,
            "evidence_excerpt": ev,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        candidates.append(candidate)
    return candidates

# -------------------- Utility: write cluster files (full entity objects) -----------
def write_cluster_summary(path: Path, cluster_map: Dict[int, List[int]], entities: List[Dict]):
    n_entities = len(entities)
    clusters = {}
    for k, idxs in sorted(cluster_map.items(), key=lambda x: x[0]):
        arr = []
        for i in idxs:
            ent = entities[i]
            arr.append(ent)
        clusters[str(k)] = arr
    meta = {"n_entities": n_entities, "n_clusters": len(clusters), "clusters": clusters}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')
        cluster_items = list(clusters.items())
        for ci, (k, ents) in enumerate(cluster_items):
            fh.write(f'    "{k}": [\n')
            for ei, ent in enumerate(ents):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'      {ent_json}')
                if ei < len(ents) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('    ]')
            if ci < len(cluster_items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

def write_classes_round(path: Path, candidates: List[Dict], entities: List[Dict], id_to_index: Dict[str,int]):
    """
    Write classes in an enriched cluster-like visual form:
    {
      "n_classes": <int>,
      "n_members_total": <int>,
      "classes": {
         "<candidate_id>": {
             "class_label": "...",
             "class_description": "...",
             "confidence": 0.9,
             "evidence_excerpt": "...",
             "source_cluster_id": <int|null>,
             "members": [ <full entity obj>, ... ]
         },
         ...
      }
    }
    """
    classes_map = {}
    total_members = 0
    for c in candidates:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        mids = c.get("member_ids", [])
        member_objs = []
        for mid in mids:
            if mid in id_to_index:
                member_objs.append(entities[id_to_index[mid]])
        if not member_objs:
            continue
        # gather metadata (ensure keys exist)
        meta = {
            "class_label": c.get("class_label", ""),
            "class_description": c.get("class_description", ""),
            "confidence": float(c.get("confidence", 0.0)),
            "evidence_excerpt": c.get("evidence_excerpt", ""),
            "source_cluster_id": c.get("source_cluster_id", None),
            "members": member_objs
        }
        classes_map[cid] = meta
        total_members += len(member_objs)
    meta = {"n_classes": len(classes_map), "n_members_total": total_members, "classes": classes_map}
    # write in requested visual format: pretty top-level, compact entity lines
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_classes": {meta["n_classes"]},\n')
        fh.write(f'  "n_members_total": {meta["n_members_total"]},\n')
        fh.write('  "classes": {\n')
        items = list(classes_map.items())
        for ci, (k, cls_meta) in enumerate(items):
            fh.write(f'    "{k}": {{\n')
            # write metadata fields
            fh.write(f'      "class_label": {json.dumps(cls_meta["class_label"], ensure_ascii=False)},\n')
            fh.write(f'      "class_description": {json.dumps(cls_meta["class_description"], ensure_ascii=False)},\n')
            fh.write(f'      "confidence": {json.dumps(cls_meta["confidence"], ensure_ascii=False)},\n')
            fh.write(f'      "evidence_excerpt": {json.dumps(cls_meta["evidence_excerpt"], ensure_ascii=False)},\n')
            fh.write(f'      "source_cluster_id": {json.dumps(cls_meta["source_cluster_id"], ensure_ascii=False)},\n')
            fh.write(f'      "members": [\n')
            for ei, ent in enumerate(cls_meta["members"]):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'        {ent_json}')
                if ei < len(cls_meta["members"]) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('      ]\n')
            fh.write('    }')
            if ci < len(items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main():
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    # ensure ids exist
    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    # id -> index
    id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    # prepare embedder and embeddings
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    # initial coarse clustering
    labels, _ = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    # cluster membership mapping
    cluster_to_indices = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    # save initial clustering with full entity objects
    write_cluster_summary(INITIAL_CLUSTER_OUT, cluster_to_indices, entities)
    if VERBOSE: print(f"[write] initial cluster file -> {INITIAL_CLUSTER_OUT}")

    # bookkeeping sets
    seen_by_llm = set()         # entity ids that were passed to LLM at least once
    assigned_entity_ids = set() # entity ids assigned into class candidates
    all_candidates = []         # cumulative across rounds (to write class_candidates.jsonl)

    # local helper to call LLM on member list and return candidates (and mark seen/assigned)
    def call_and_record(members_indices: List[int], source_cluster: Optional[int]=None, single_entity_mode: bool=False) -> List[Dict]:
        nonlocal seen_by_llm, assigned_entity_ids, all_candidates
        if not members_indices:
            return []
        members = [entities[i] for i in members_indices]
        # call LLM in chunks if large
        results = []
        for i in range(0, len(members), MAX_MEMBERS_PER_PROMPT):
            chunk = members[i:i+MAX_MEMBERS_PER_PROMPT]
            chunk_indices = members_indices[i:i+MAX_MEMBERS_PER_PROMPT]
            # mark seen
            for m in chunk:
                seen_by_llm.add(m["id"])
            candidates = process_member_chunk_llm(chunk, single_entity_mode=single_entity_mode)
            # attach member_entities, source_cluster and record assigned ids
            for c in candidates:
                mids = c.get("member_ids", [])
                member_entities = [entities[id_to_index[mid]] for mid in mids if mid in id_to_index]
                if not member_entities:
                    continue
                # attach full members and source cluster id for traceability
                c["member_entities"] = member_entities
                c["source_cluster_id"] = source_cluster
                all_candidates.append(c)
                for mid in mids:
                    assigned_entity_ids.add(mid)
                results.append(c)
        return results

    # ---------- Round 0: initial pass over non-noise coarse clusters ----------
    round0_candidates = []
    if VERBOSE: print("[round0] processing coarse non-noise clusters")
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE: print(f"[round0] cluster {lab} size={len(idxs)}")
        # if cluster large, do local subcluster; else call directly
        if len(idxs) > MAX_CLUSTER_SIZE_FOR_LOCAL:
            # local subcluster
            try:
                sub_emb = combined_emb[idxs]
                local_clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                                                  min_samples=LOCAL_HDBSCAN_MIN_SAMPLES, metric='euclidean', cluster_selection_method='eom')
                local_labels = local_clusterer.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(idxs), dtype=int)
            local_map = {}
            for i_local, lab_local in enumerate(local_labels):
                global_idx = idxs[i_local]
                local_map.setdefault(int(lab_local), []).append(global_idx)
            for sublab, subidxs in local_map.items():
                if sublab == -1:
                    continue
                # source cluster id set to coarse cluster `lab` and sublab appended as tuple (lab, sublab)
                source_id = {"coarse_cluster": lab, "local_subcluster": int(sublab)}
                cand = call_and_record(subidxs, source_cluster=source_id, single_entity_mode=(len(subidxs) == 1))
                round0_candidates.extend(cand)
        else:
            source_id = {"coarse_cluster": lab, "local_subcluster": None}
            cand = call_and_record(idxs, source_cluster=source_id, single_entity_mode=(len(idxs) == 1))
            round0_candidates.extend(cand)

    # save classes for round 0
    classes_round0_path = Path(f"{CLASSES_PREFIX}0.json")
    write_classes_round(classes_round0_path, round0_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round 0 -> {classes_round0_path}")

    # ---------- Iterative recluster rounds ----------
    # initial pool: original noise (cluster -1) + seen but unassigned
    original_noise_indices = cluster_to_indices.get(-1, [])
    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        # pool composition: original noise + seen-but-unassigned
        seen_but_unassigned = list(seen_by_llm - assigned_entity_ids)
        pool_ids = {entities[i]["id"] for i in original_noise_indices}
        pool_ids.update(seen_but_unassigned)
        # remove already assigned
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE: print(f"[reclust {round_num}] pool empty -> stopping")
            break
        pool_indices = [id_to_index[pid] for pid in pool_ids if pid in id_to_index]
        if not pool_indices:
            if VERBOSE: print(f"[reclust {round_num}] no valid pool indices -> stopping")
            break

        if VERBOSE: print(f"[reclust {round_num}] reclustering pool size={len(pool_indices)}")
        # recluster pool (conservative min size 2)
        try:
            sub_emb = combined_emb[pool_indices]
            labels_sub, _ = run_hdbscan(sub_emb, min_cluster_size=2, min_samples=1, use_umap=False)
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        # build map label->global indices
        sub_cluster_map = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        # save recluster summary
        recluster_path = Path(f"{RECLUSTER_PREFIX}{round_num}.json")
        write_cluster_summary(recluster_path, sub_cluster_map, entities)
        if VERBOSE: print(f"[write] recluster round {round_num} -> {recluster_path}")

        # process each non-noise subcluster and collect classes for this round
        round_candidates = []
        new_classes_count = 0
        for lab_sub, gidxs in sorted(sub_cluster_map.items(), key=lambda x: (x[0]==-1, x[0])):
            if lab_sub == -1:
                continue
            if VERBOSE: print(f"[reclust {round_num}] processing subcluster {lab_sub} size={len(gidxs)}")
            # source cluster id set to recluster round + subcluster label
            source_id = {"recluster_round": round_num, "subcluster": int(lab_sub)}
            cand = call_and_record(gidxs, source_cluster=source_id, single_entity_mode=(len(gidxs) == 1))
            round_candidates.extend(cand)
            new_classes_count += len(cand)

        # save classes found this round in enriched format
        classes_round_path = Path(f"{CLASSES_PREFIX}{round_num}.json")
        write_classes_round(classes_round_path, round_candidates, entities, id_to_index)
        if VERBOSE: print(f"[write] classes round {round_num} -> {classes_round_path}  (new_classes={new_classes_count})")

        if new_classes_count == 0:
            if VERBOSE: print(f"[reclust {round_num}] no new classes -> stopping recluster loop")
            break
        # otherwise continue to next recluster round (pool will be recomputed)

    # ---------- Final single-entity pass ----------
    # remaining after all reclustering rounds: entities not assigned yet
    remaining_after_reclustering = [e for e in entities if e["id"] not in assigned_entity_ids]
    if VERBOSE: print(f"[single pass] remaining entities (before single-entity pass): {len(remaining_after_reclustering)}")

    single_candidates = []
    for e in remaining_after_reclustering:
        source_id = {"single_pass": True}
        cand = call_and_record([id_to_index[e["id"]]], source_cluster=source_id, single_entity_mode=True)
        single_candidates.extend(cand)

    # save classes from single-entity pass
    classes_single_path = Path(f"{CLASSES_PREFIX}single.json")
    write_classes_round(classes_single_path, single_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round single -> {classes_single_path}")

    # ---------- Write cumulative class candidates file (line-per-candidate) ----------
    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in all_candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] cumulative class_candidates -> {CLASS_CANDIDATES_OUT} (count={len(all_candidates)})")

    # ---------- Final remaining entities (never assigned after full pipeline) ----------
    final_remaining = [e for e in entities if e["id"] not in assigned_entity_ids]
    with open(REMAINING_OUT, "w", encoding="utf-8") as fh:
        for e in final_remaining:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] final remaining entities -> {REMAINING_OUT} (count={len(final_remaining)})")

    print("[done] ClassRec iterative v3 finished.")

if __name__ == "__main__":
    classrec_iterative_main()



#endregion#? Cls Rec V3
#*#########################  End  ##########################





#?######################### Start ##########################
#region:#?   Cls Rec V4 - Class hint type included

#!/usr/bin/env python3
"""
classrec_iterative_v4.py

Iterative Class Recognition (ClassRec) with per-iteration class outputs
matching the cluster-file visual/JSON format and including class metadata
(label/desc/confidence/evidence + source cluster id + class_type_hint).

This is a fix for KeyError caused by unescaped braces in the prompt template.
All literal braces in the prompt are escaped ({{ and }}), except {members_block}.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CANDIDATES_OUT = OUT_DIR / "class_candidates.jsonl"
INITIAL_CLUSTER_OUT = OUT_DIR / "initial_cluster_entities.json"
RECLUSTER_PREFIX = OUT_DIR / "recluster_round_"
CLASSES_PREFIX = OUT_DIR / "classes_round_"
REMAINING_OUT = OUT_DIR / "remaining_entities.jsonl"

# embedder / model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}

# HDBSCAN + UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0

# local subcluster
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# prompt and LLM / limits
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 2500

# iteration control
MAX_RECLUSTER_ROUNDS = 12  # safety cap
VERBOSE = True

# ------------------------ OpenAI client loader -----------------------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ------------------------- HF Embedder ------------------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ----------------------------------
def load_entities(path: Path) -> List[Dict]:
    assert path.exists(), f"Input not found: {path}"
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ents.append(json.loads(line))
    return ents

def safe_text(e: Dict, k: str) -> str:
    v = e.get(k)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or "")
        descs.append(safe_text(e, "entity_description") or "")
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or ""
        et = safe_text(e, "entity_type_hint") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
            if pieces:
                node_props_text = " | ".join(pieces)
        parts = []
        if et:
            parts.append(f"[TYPE:{et}]")
        if resolution:
            parts.append(resolution)
        if node_props_text:
            parts.append(node_props_text)
        ctxs.append(" ; ".join(parts))
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name); emb_desc = _ensure(emb_desc); emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0); w_desc = weights.get("desc", 0.0); w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0: raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0]-1)),
                            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0]-1)),
                            min_dist=UMAP_MIN_DIST,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ------------------ Prompt (REVISED to be MUST) -----------------------
# NOTE: all literal braces are escaped ({{ and }}) except {members_block}
CLASS_PROMPT_TEMPLATE = """
You are a careful schema / ontology suggester.
Your job is to induce a meaningful, reusable schema from entity-level evidence.
Be conservative, precise, and resist over-generalization.

=====================
INPUT YOU ARE GIVEN
=====================

Each entity you see is already a resolved entity from a previous pipeline stage.
For each entity, you are given the following fields:

- entity_id: (string) unique id for the entity.
- entity_name: (string) short canonical entity-level name.
- entity_description: (string) concise explanation of the entity.
- resolution_context: (string) a 20–120 word excerpt explaining why this entity was named this way;
  this is the PRIMARY semantic evidence.
- entity_type_hint: (string) a weak, local hint about entity role (e.g., Material, Component, FailureMechanism).
  This is a suggestion only and may be incorrect.
- node_properties: (optional) small list/dict of intrinsic properties (e.g., grade:304).

=====================
YOUR TASK
=====================

Given a small group of entities, suggest ZERO or more classes that group entities
which genuinely belong together in a practical, reusable way.

Important rules (ENFORCED):
- If the input contains MULTIPLE entities:
  -> You MUST ONLY create classes that contain TWO OR MORE members.
  -> You MUST NOT create a single-member class when multiple entities are present.
- The ONLY situation where a single-member class is allowed:
  -> When the input contains EXACTLY ONE entity (single-entity mode).

Do NOT force entities into classes by broadening or renaming a class to "fit" them.
If an entity does not clearly belong, omit it — it will be revisited later.

SINGLE-ENTITY MODE (IMPORTANT)
- You are receiving EXACTLY ONE entity in this prompt. In this case you SHOULD produce exactly one class that contains that entity (a single-member class). 
- The single-member class must include: class_label, class_description, class_type_hint (if possible), member_ids (use the provided entity_id), and a confidence value.  
- Do NOT invent other entity_ids; use the entity_id exactly as provided. If you judge that no sensible class exists, still return a short single-member class using a conservative label like "Misc: <entity_name>" with low confidence (e.g., 0.10) rather than returning an empty array. This helps downstream experiments while keeping the class low-weight.


=====================
TWO-LEVEL SCHEMA
=====================

We are building a TWO-LEVEL schema:

Level 1 (Classes): groups of entities (e.g., "High-Temperature Corrosion")
Level 2 (Class_Type_Hint): an upper-level connector that groups classes (e.g., "Failure Mechanism")

- Class_Type_Hint is NOT the same as entity_type_hint.
- Infer Class_Type_Hint from the class members; do NOT blindly copy entity_type_hint.

=====================
OUTPUT FORMAT (REQUIRED)
=====================

Return ONLY a JSON ARRAY.

Each element must have:
- class_label (string): short canonical name (1-3 words)
- class_description (string): 1–2 sentences explaining membership & distinction
- class_type_hint (string): upper-level family (e.g., "Failure Mechanism")
- member_ids (array[string]): entity_ids from the input that belong to this class
- confidence (float): 0.0–1.0 confidence estimate
- evidence_excerpt (string, optional): brief excerpt (5–30 words) that supports the grouping

HARD OUTPUT RULES:
- Use ONLY provided entity_ids.
- member_ids MUST be from the input.
- Prefer non-overlapping classes; small overlap allowed only if justified.
- If no sensible class, return [].

=====================
EXAMPLES
=====================

GOOD:
Input entities:
- graphitization (En_1)
- sulfidation    (En_2)

Output:
[
  {{
    "class_label": "High-Temperature Degradation",
    "class_description": "Material degradation mechanisms at elevated temperatures (graphitization, sulfidation).",
    "class_type_hint": "Failure Mechanism",
    "member_ids": ["En_1","En_2"],
    "confidence": 0.87
  }}
]

BAD (DO NOT DO):
Entities: graphitization, pressure gauge
-> Do NOT output a broad "Equipment Issue" that forces both into one class. Prefer [].

=====================
ENTITIES
=====================

Each entity below is provided as:
- entity_id
- entity_name
- entity_description
- resolution_context
- entity_type_hint
- node_properties

Entities:
{members_block}

Return JSON array only.
"""

def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id", "")
        name = (m.get("entity_name") or "")[:120].replace("\n", " ")
        desc = (m.get("entity_description") or "")[:300].replace("\n", " ")
        res = (m.get("resolution_context") or m.get("context_phrase") or "")[:400].replace("\n", " ")
        et = (m.get("entity_type_hint") or "")[:80].replace("\n", " ")
        node_props = m.get("node_properties") or []
        np_txt = json.dumps(node_props, ensure_ascii=False) if node_props else ""
        rows.append(f"{eid} | {name} | {desc} | {res} | {et} | {np_txt}")
    return "\n".join(rows)

def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    s = txt.strip()
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(members: List[Dict], single_entity_mode: bool = False) -> List[Dict]:
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE: print(f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}")
        return []
    llm_out = call_llm(prompt)
    arr = parse_json_array_from_text(llm_out)
    if not arr:
        return []
    candidates = []
    provided_ids = {m.get("id") for m in members}
    for c in arr:
        label = c.get("class_label") or c.get("label") or c.get("name")
        if not label:
            continue
        member_ids = c.get("member_ids") or c.get("members") or []
        member_ids = [mid for mid in member_ids if mid in provided_ids]
        if not member_ids:
            continue
        if not single_entity_mode and len(members) > 1 and len(member_ids) < 2:
            continue
        confidence = float(c.get("confidence", 0.0)) if c.get("confidence") is not None else 0.0
        desc = c.get("class_description") or c.get("description") or ""
        ev = c.get("evidence_excerpt") or ""
        class_type = c.get("class_type_hint") or c.get("class_type") or ""
        candidate = {
            "candidate_id": "ClsC_" + uuid.uuid4().hex[:8],
            "class_label": label,
            "class_description": desc,
            "class_type_hint": class_type,
            "member_ids": member_ids,
            "confidence": confidence,
            "evidence_excerpt": ev,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        candidates.append(candidate)
    return candidates

# -------------------- Utility: write cluster files (full entity objects) -----------
def write_cluster_summary(path: Path, cluster_map: Dict[int, List[int]], entities: List[Dict]):
    n_entities = len(entities)
    clusters = {}
    for k, idxs in sorted(cluster_map.items(), key=lambda x: x[0]):
        arr = []
        for i in idxs:
            ent = entities[i]
            arr.append(ent)
        clusters[str(k)] = arr
    meta = {"n_entities": n_entities, "n_clusters": len(clusters), "clusters": clusters}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')
        cluster_items = list(clusters.items())
        for ci, (k, ents) in enumerate(cluster_items):
            fh.write(f'    "{k}": [\n')
            for ei, ent in enumerate(ents):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'      {ent_json}')
                if ei < len(ents) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('    ]')
            if ci < len(cluster_items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

def write_classes_round(path: Path, candidates: List[Dict], entities: List[Dict], id_to_index: Dict[str,int]):
    classes_map = {}
    total_members = 0
    for c in candidates:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        mids = c.get("member_ids", [])
        member_objs = []
        for mid in mids:
            if mid in id_to_index:
                member_objs.append(entities[id_to_index[mid]])
        if not member_objs:
            continue
        meta = {
            "class_label": c.get("class_label", ""),
            "class_description": c.get("class_description", ""),
            "class_type_hint": c.get("class_type_hint", ""),
            "confidence": float(c.get("confidence", 0.0)),
            "evidence_excerpt": c.get("evidence_excerpt", ""),
            "source_cluster_id": c.get("source_cluster_id", None),
            "members": member_objs
        }
        classes_map[cid] = meta
        total_members += len(member_objs)
    meta = {"n_classes": len(classes_map), "n_members_total": total_members, "classes": classes_map}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_classes": {meta["n_classes"]},\n')
        fh.write(f'  "n_members_total": {meta["n_members_total"]},\n')
        fh.write('  "classes": {\n')
        items = list(classes_map.items())
        for ci, (k, cls_meta) in enumerate(items):
            fh.write(f'    "{k}": {{\n')
            fh.write(f'      "class_label": {json.dumps(cls_meta["class_label"], ensure_ascii=False)},\n')
            fh.write(f'      "class_description": {json.dumps(cls_meta["class_description"], ensure_ascii=False)},\n')
            fh.write(f'      "class_type_hint": {json.dumps(cls_meta["class_type_hint"], ensure_ascii=False)},\n')
            fh.write(f'      "confidence": {json.dumps(cls_meta["confidence"], ensure_ascii=False)},\n')
            fh.write(f'      "evidence_excerpt": {json.dumps(cls_meta["evidence_excerpt"], ensure_ascii=False)},\n')
            fh.write(f'      "source_cluster_id": {json.dumps(cls_meta["source_cluster_id"], ensure_ascii=False)},\n')
            fh.write(f'      "members": [\n')
            for ei, ent in enumerate(cls_meta["members"]):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'        {ent_json}')
                if ei < len(cls_meta["members"]) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('      ]\n')
            fh.write('    }')
            if ci < len(items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main():
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    labels, _ = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    cluster_to_indices = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    write_cluster_summary(INITIAL_CLUSTER_OUT, cluster_to_indices, entities)
    if VERBOSE: print(f"[write] initial cluster file -> {INITIAL_CLUSTER_OUT}")

    seen_by_llm = set()
    assigned_entity_ids = set()
    all_candidates = []

    def call_and_record(members_indices: List[int], source_cluster: Optional[object]=None, single_entity_mode: bool=False) -> List[Dict]:
        nonlocal seen_by_llm, assigned_entity_ids, all_candidates
        if not members_indices:
            return []
        members = [entities[i] for i in members_indices]
        results = []
        for i in range(0, len(members), MAX_MEMBERS_PER_PROMPT):
            chunk = members[i:i+MAX_MEMBERS_PER_PROMPT]
            for m in chunk:
                if m.get("id"):
                    seen_by_llm.add(m["id"])
            candidates = process_member_chunk_llm(chunk, single_entity_mode=single_entity_mode)
            for c in candidates:
                mids = c.get("member_ids", [])
                member_entities = [entities[id_to_index[mid]] for mid in mids if mid in id_to_index]
                if not member_entities:
                    continue
                c["member_entities"] = member_entities
                c["source_cluster_id"] = source_cluster
                all_candidates.append(c)
                for mid in mids:
                    assigned_entity_ids.add(mid)
                results.append(c)
        return results

    round0_candidates = []
    if VERBOSE: print("[round0] processing coarse non-noise clusters")
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE: print(f"[round0] cluster {lab} size={len(idxs)}")
        if len(idxs) > MAX_CLUSTER_SIZE_FOR_LOCAL:
            try:
                sub_emb = combined_emb[idxs]
                local_clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                                                  min_samples=LOCAL_HDBSCAN_MIN_SAMPLES, metric='euclidean', cluster_selection_method='eom')
                local_labels = local_clusterer.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(idxs), dtype=int)
            local_map = {}
            for i_local, lab_local in enumerate(local_labels):
                global_idx = idxs[i_local]
                local_map.setdefault(int(lab_local), []).append(global_idx)
            for sublab, subidxs in local_map.items():
                if sublab == -1:
                    continue
                source_id = {"coarse_cluster": int(lab), "local_subcluster": int(sublab)}
                cand = call_and_record(subidxs, source_cluster=source_id, single_entity_mode=(len(subidxs) == 1))
                round0_candidates.extend(cand)
        else:
            source_id = {"coarse_cluster": int(lab), "local_subcluster": None}
            cand = call_and_record(idxs, source_cluster=source_id, single_entity_mode=(len(idxs) == 1))
            round0_candidates.extend(cand)

    classes_round0_path = Path(f"{CLASSES_PREFIX}0.json")
    write_classes_round(classes_round0_path, round0_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round 0 -> {classes_round0_path}")

    original_noise_indices = cluster_to_indices.get(-1, [])
    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        seen_but_unassigned = list(seen_by_llm - assigned_entity_ids)
        pool_ids = {entities[i]["id"] for i in original_noise_indices}
        pool_ids.update(seen_but_unassigned)
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE: print(f"[reclust {round_num}] pool empty -> stopping")
            break
        pool_indices = [id_to_index[pid] for pid in pool_ids if pid in id_to_index]
        if not pool_indices:
            if VERBOSE: print(f"[reclust {round_num}] no valid pool indices -> stopping")
            break

        if VERBOSE: print(f"[reclust {round_num}] reclustering pool size={len(pool_indices)}")
        try:
            sub_emb = combined_emb[pool_indices]
            labels_sub, _ = run_hdbscan(sub_emb, min_cluster_size=2, min_samples=1, use_umap=False)
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        sub_cluster_map = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        recluster_path = Path(f"{RECLUSTER_PREFIX}{round_num}.json")
        write_cluster_summary(recluster_path, sub_cluster_map, entities)
        if VERBOSE: print(f"[write] recluster round {round_num} -> {recluster_path}")

        round_candidates = []
        new_classes_count = 0
        for lab_sub, gidxs in sorted(sub_cluster_map.items(), key=lambda x: (x[0]==-1, x[0])):
            if lab_sub == -1:
                continue
            if VERBOSE: print(f"[reclust {round_num}] processing subcluster {lab_sub} size={len(gidxs)}")
            source_id = {"recluster_round": int(round_num), "subcluster": int(lab_sub)}
            cand = call_and_record(gidxs, source_cluster=source_id, single_entity_mode=(len(gidxs) == 1))
            round_candidates.extend(cand)
            new_classes_count += len(cand)

        classes_round_path = Path(f"{CLASSES_PREFIX}{round_num}.json")
        write_classes_round(classes_round_path, round_candidates, entities, id_to_index)
        if VERBOSE: print(f"[write] classes round {round_num} -> {classes_round_path}  (new_classes={new_classes_count})")

        if new_classes_count == 0:
            if VERBOSE: print(f"[reclust {round_num}] no new classes -> stopping recluster loop")
            break

    remaining_after_reclustering = [e for e in entities if e["id"] not in assigned_entity_ids]
    if VERBOSE: print(f"[single pass] remaining entities (before single-entity pass): {len(remaining_after_reclustering)}")

    single_candidates = []
    for e in remaining_after_reclustering:
        source_id = {"single_pass": True}
        cand = call_and_record([id_to_index[e["id"]]], source_cluster=source_id, single_entity_mode=True)
        single_candidates.extend(cand)

    classes_single_path = Path(f"{CLASSES_PREFIX}single.json")
    write_classes_round(classes_single_path, single_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round single -> {classes_single_path}")

    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in all_candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] cumulative class_candidates -> {CLASS_CANDIDATES_OUT} (count={len(all_candidates)})")

    final_remaining = [e for e in entities if e["id"] not in assigned_entity_ids]
    with open(REMAINING_OUT, "w", encoding="utf-8") as fh:
        for e in final_remaining:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] final remaining entities -> {REMAINING_OUT} (count={len(final_remaining)})")

    print("[done] ClassRec iterative v4 finished.")

if __name__ == "__main__":
    classrec_iterative_main()


#endregion#? Cls Rec V4 - Class hint type included
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Create input for Cls Res from per-round classes




#!/usr/bin/env python3
"""
merge_classes_for_cls_res.py

Merge per-round classes files (classes_round_*.json) into a single
JSONL + JSON file suitable as input to the next step (Cls Res).

Output:
 - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.jsonl
 - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json
"""

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec")
PATTERN = "classes_round_*.json"
OUT_JSONL = ROOT / "classes_for_cls_res.jsonl"
OUT_JSON  = ROOT / "classes_for_cls_res.json"

def read_classes_file(p: Path):
    """
    Reads a classes_round file in the format produced by write_classes_round.
    Returns dict candidate_id -> class_meta (with members list of full entity objects)
    """
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to parse {p}: {e}")
        return {}
    classes = j.get("classes") or {}
    out = {}
    for cid, meta in classes.items():
        # meta expected to contain class_label, class_description, class_type_hint, confidence, evidence_excerpt, source_cluster_id, members
        # ensure members is list
        members = meta.get("members") or []
        # normalize member objects: ensure each has an "id"
        members_norm = []
        for m in members:
            if isinstance(m, dict) and "id" in m:
                members_norm.append(m)
            else:
                # skip bad members
                continue
        out[cid] = {
            "candidate_id": cid,
            "class_label": meta.get("class_label", "") or "",
            "class_description": meta.get("class_description", "") or "",
            "class_type_hint": meta.get("class_type_hint", "") or "",
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "evidence_excerpt": meta.get("evidence_excerpt", "") or "",
            "source_cluster_id": meta.get("source_cluster_id", None),
            "members": members_norm,
            # provenance: where we loaded it from
            "_source_file": str(p)
        }
    return out

def merge_classes(all_classes_by_file):
    """
    Merge classes by (label, class_type_hint) normalized key.
    If same key seen multiple times:
      - union members (unique by id)
      - take class_description from the instance with highest confidence (tie -> latest file appearance)
      - confidence = max(confidences)
      - evidence_excerpt: prefer higher confidence's excerpt
      - source_files: list
      - candidate_ids: list of contributing candidate ids
    """
    merged = {}
    key_to_ids = defaultdict(list)  # for debugging
    for cid, c in all_classes_by_file.items():
        key = (c["class_label"].strip().lower(), (c.get("class_type_hint") or "").strip().lower())
        if key not in merged:
            merged[key] = {
                "canonical_class_label": c["class_label"],
                "class_type_hint": c.get("class_type_hint", ""),
                "class_description": c.get("class_description", ""),
                "confidence": c.get("confidence", 0.0),
                "evidence_excerpt": c.get("evidence_excerpt", "") or "",
                "members_by_id": {m["id"]: m for m in c.get("members", [])},
                "candidate_ids": [c["candidate_id"]],
                "source_files": [c.get("_source_file")],
            }
        else:
            cur = merged[key]
            # union members
            for m in c.get("members", []):
                cur["members_by_id"][m["id"]] = m
            # update confidence & description/evidence if this c has higher confidence
            if c.get("confidence", 0.0) > cur["confidence"]:
                cur["confidence"] = c.get("confidence", 0.0)
                # replace description/evidence with higher-confidence one
                if c.get("class_description"):
                    cur["class_description"] = c.get("class_description", cur["class_description"])
                if c.get("evidence_excerpt"):
                    cur["evidence_excerpt"] = c.get("evidence_excerpt", cur["evidence_excerpt"])
            # append provenance
            cur["candidate_ids"].append(c["candidate_id"])
            sf = c.get("_source_file")
            if sf and sf not in cur["source_files"]:
                cur["source_files"].append(sf)
        key_to_ids[key].append(cid)

    # convert merged to output list
    out_list = []
    for key, v in merged.items():
        members = list(v["members_by_id"].values())
        out_obj = {
            "class_label": v["canonical_class_label"],
            "class_type_hint": v["class_type_hint"],
            "class_description": v.get("class_description",""),
            "confidence": float(v.get("confidence",0.0)),
            "evidence_excerpt": v.get("evidence_excerpt",""),
            "member_ids": [m["id"] for m in members],
            "members": members,
            "candidate_ids": v.get("candidate_ids", []),
            "source_files": v.get("source_files", [])
        }
        out_list.append(out_obj)
    return out_list

def main():
    files = sorted(ROOT.glob(PATTERN))
    if not files:
        print(f"[error] no files found matching {PATTERN} in {ROOT}")
        return
    print(f"[info] found {len(files)} files. Reading...")
    all_classes = {}
    total_classes = 0
    for f in files:
        cls_map = read_classes_file(f)
        if not cls_map:
            continue
        for cid, c in cls_map.items():
            # ensure candidate id unique by prefixing file if collided
            if cid in all_classes:
                cid_new = f"{Path(c['_source_file']).stem}__{cid}"
                c["candidate_id"] = cid_new
                all_classes[cid_new] = c
            else:
                all_classes[cid] = c
        total_classes += len(cls_map)
    print(f"[info] collected {len(all_classes)} raw class entries (from {total_classes} source classes). Merging...")

    merged = merge_classes(all_classes)
    print(f"[info] merged -> {len(merged)} classes. Writing outputs...")

    # write JSONL (one merged class per line) and JSON (array)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh_jsonl, open(OUT_JSON, "w", encoding="utf-8") as fh_json:
        for c in merged:
            fh_jsonl.write(json.dumps(c, ensure_ascii=False) + "\n")
        fh_json.write(json.dumps(merged, ensure_ascii=False, indent=2))

    # summary
    total_members = sum(len(c.get("members", [])) for c in merged)
    print(f"[done] wrote {OUT_JSONL} and {OUT_JSON}.")
    print(f"       classes: {len(merged)}, total member objects (post-merge, possibly duplicated across classes): {total_members}")

if __name__ == "__main__":
    main()


#endregion#? Create input for Cls Res from per-round classes
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Cls Res - Class Resolution


#!/usr/bin/env python3
"""
class_resolution.py

Class Resolution (Cls Res)

- Loads per-round classes JSON produced by ClassRec (e.g., classes_round_0.json).
- Loads entity inputs (cls_input_entities.jsonl) to have entity metadata available.
- Embeds class candidates (class_label, class_description, class_type_hint, evidence_excerpt)
  plus compact member summaries (id/name/desc/type) folded into ctx.
- Clusters class candidates with HDBSCAN (optional UMAP).
- Iterates clusters (exclude -1 noise); for each cluster:
    - sends a JSON prompt to the LLM asking it to return an ORDERED list of function calls (Merge/Create/Reassign/Modify),
    - parses the JSON response and executes each function in order locally,
    - writes canonical/merged classes and a resolution log.
- Writes outputs to CLASSES_RES_DIR.

Author: assistant (based on your pipeline)
"""
import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---- ML/embedding libs (same style as other scripts) ----
import numpy as np
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required - pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# ---- OpenAI client loader (reuse your pattern) ----
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI API key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

# ---- Paths (edit if needed) ----
CLASSES_ROUND_FILE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_round_0.json")
CLS_INPUT_ENTITIES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
CLASSES_RES_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
CLASSES_RES_DIR.mkdir(parents=True, exist_ok=True)
OUT_CLASSES_JSONL = CLASSES_RES_DIR / "classes_resolved.jsonl"
OUT_CANONICAL_JSONL = CLASSES_RES_DIR / "canonical_classes.jsonl"
OUT_LOG_JSONL = CLASSES_RES_DIR / "cls_res_log.jsonl"

# ---- Embedding / clustering config (tweakable) ----
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # same default you used elsewhere
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"label": 0.35, "desc": 0.25, "evidence": 0.10, "members": 0.30}  # tuneable
USE_UMAP = True
UMAP_DIMS = 64
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# ---- LLM / prompt params ----
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1000
PROMPT_CHAR_PER_TOKEN = 4
MAX_PROMPT_TOKENS_EST = 2500
TRUNC_FIELD = 400  # truncate long descriptions/evidence to keep prompt compact

# ---- Helpers: file loaders ----
def load_json(path: Path):
    j = json.loads(path.read_text(encoding="utf-8"))
    return j

def load_jsonl(path: Path) -> List[Dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

# ---- HF embedder (same pattern) ----
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---- Build class text fields for embedding ----
def safe_truncate(s, n=TRUNC_FIELD):
    if not s:
        return ""
    s = str(s).replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "..."

def build_class_field_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Build lists:
      - label_texts
      - desc_texts
      - evidence_texts
      - members_texts (combined short summaries of members)
    """
    labels, descs, evids, members = [], [], [], []
    for c in classes:
        labels.append(safe_truncate(c.get("class_label",""), 120))
        descs.append(safe_truncate(c.get("class_description",""), TRUNC_FIELD))
        evids.append(safe_truncate(c.get("evidence_excerpt",""), TRUNC_FIELD))
        # build compact members text: id:name:shortdesc:type; join with " | "
        mems = c.get("members", []) or []
        pieces = []
        for m in mems:
            mid = m.get("id","")
            mname = safe_truncate(m.get("entity_name",""), 50)
            mdesc = safe_truncate(m.get("entity_description",""), 120)
            mtype = safe_truncate(m.get("entity_type_hint",""), 30)
            piece = f"{mid}:{mname}"
            if mdesc:
                piece += f" ({mdesc})"
            if mtype:
                piece += f" [{mtype}]"
            pieces.append(piece)
        members_text = " | ".join(pieces) if pieces else ""
        members.append(safe_truncate(members_text, TRUNC_FIELD))
    return labels, descs, evids, members

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights=WEIGHTS) -> np.ndarray:
    labels, descs, evids, members = build_class_field_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_evid = embedder.encode_batch(evids) if any(t.strip() for t in evids) else None
    emb_mems = embedder.encode_batch(members) if any(t.strip() for t in members) else None

    # find D
    D = None
    for arr in (emb_label, emb_desc, emb_evid, emb_mems):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = _ensure(emb_label); emb_desc = _ensure(emb_desc); emb_evid = _ensure(emb_evid); emb_mems = _ensure(emb_mems)

    w_label = weights.get("label", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_evid = weights.get("evidence", 0.0)
    w_mems = weights.get("members", 0.0)
    Wsum = w_label + w_desc + w_evid + w_mems
    if Wsum <= 0:
        raise ValueError("invalid weights sum")
    w_label /= Wsum; w_desc /= Wsum; w_evid /= Wsum; w_mems /= Wsum

    combined = (w_label * emb_label) + (w_desc * emb_desc) + (w_evid * emb_evid) + (w_mems * emb_mems)
    combined = normalize(combined, axis=1)
    return combined

# ---- clustering helper ----
def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP):
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_DIMS, max(2, X.shape[0]-1)),
                            n_neighbors=min(15, max(2, X.shape[0]-1)),
                            min_dist=0.0,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=HDBSCAN_METRIC, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---- LLM call & robust JSON extraction ----
def call_llm(prompt: str, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_from_text(text: str):
    """Try to extract the first JSON array or object from model text."""
    if not text:
        return None
    s = text.strip()
    # remove fenced codeblocks
    if s.startswith("```"):
        s = s.strip("`")
    # find first '[' and matching ']' or '{' and matching '}'
    idx_arr_start = s.find("[")
    idx_obj_start = s.find("{")
    candidate = None
    if idx_arr_start != -1:
        # heuristically take last ']' after start
        idx_arr_end = s.rfind("]")
        if idx_arr_end != -1 and idx_arr_end > idx_arr_start:
            candidate = s[idx_arr_start:idx_arr_end+1]
    if candidate is None and idx_obj_start != -1:
        idx_obj_end = s.rfind("}")
        if idx_obj_end != -1 and idx_obj_end > idx_obj_start:
            candidate = s[idx_obj_start:idx_obj_end+1]
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # fallback try full parse
    try:
        return json.loads(s)
    except Exception:
        return None

# ---- Allowed functions: implement them ----
def fn_merge_classes(state_classes: Dict[str, Dict], class_ids: List[str], new_name: str, new_desc: str, new_class_type_hint: str, canonical_store: List[Dict], log: List[Dict]):
    # create canonical class id
    can_id = "CanCls_" + uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    members = []
    source_ids = []
    for cid in class_ids:
        c = state_classes.get(cid)
        if not c:
            continue
        source_ids.append(cid)
        for m in c.get("members", []):
            members.append(m)
        # mark original as merged
        c["_merged_into"] = can_id
    # dedupe members by id, preserve order
    seen = set(); uniq_members = []
    for m in members:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid); uniq_members.append(m)
    canonical = {
        "canonical_id": can_id,
        "canonical_name": new_name,
        "canonical_description": new_desc,
        "canonical_type_hint": new_class_type_hint,
        "members": uniq_members,
        "source_class_ids": source_ids,
        "timestamp": timestamp,
        "source": "cls_res_merge"
    }
    canonical_store.append(canonical)
    # add canonical to state_classes as new class
    state_classes[can_id] = {
        "id": can_id,
        "class_label": new_name,
        "class_description": new_desc,
        "class_type_hint": new_class_type_hint,
        "members": uniq_members,
        "_is_canonical": True,
        "created_time": timestamp
    }
    log.append({"time": timestamp, "action": "merge_classes", "merged": source_ids, "created": can_id})
    return can_id

def fn_create_class(state_classes: Dict[str, Dict], name: str, desc: str, class_type_hint: str, member_ids: List[str], entities_index: Dict[str, Dict], log: List[Dict]):
    cid = "ClsC_" + uuid.uuid4().hex[:8]
    members = []
    for mid in member_ids or []:
        ent = entities_index.get(mid)
        if ent:
            members.append(ent)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state_classes[cid] = {
        "id": cid,
        "class_label": name,
        "class_description": desc,
        "class_type_hint": class_type_hint,
        "members": members,
        "created_time": ts
    }
    log.append({"time": ts, "action": "create_class", "created": cid, "member_ids": member_ids})
    return cid

def fn_reassign_entity(state_classes: Dict[str, Dict], entity_id: str, from_class_id: Optional[str], to_class_id: str, entities_index: Dict[str, Dict], log: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ent = entities_index.get(entity_id)
    if ent is None:
        log.append({"time": ts, "action": "reassign_entity_failed", "reason": "entity_not_found", "entity_id": entity_id})
        return False
    # remove from from_class_id if provided
    if from_class_id and from_class_id in state_classes:
        members = state_classes[from_class_id].get("members", [])
        state_classes[from_class_id]["members"] = [m for m in members if m.get("id") != entity_id]
    # add to to_class_id
    if to_class_id not in state_classes:
        # create placeholder class
        state_classes[to_class_id] = {"id": to_class_id, "class_label": to_class_id, "class_description": "", "class_type_hint": "", "members": []}
    # ensure not duplicate
    if not any(m.get("id") == entity_id for m in state_classes[to_class_id].get("members", [])):
        state_classes[to_class_id]["members"].append(ent)
    log.append({"time": ts, "action": "reassign_entity", "entity_id": entity_id, "from": from_class_id, "to": to_class_id})
    return True

def fn_modify_class(state_classes: Dict[str, Dict], class_id: str, new_name: Optional[str], new_desc: Optional[str], new_type: Optional[str], log: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c = state_classes.get(class_id)
    if not c:
        log.append({"time": ts, "action": "modify_class_failed", "reason": "class_not_found", "class_id": class_id})
        return False
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    log.append({"time": ts, "action": "modify_class", "class_id": class_id, "new_name": new_name, "new_desc": bool(new_desc), "new_type": new_type})
    return True

# ---- Orchestration: main loop ----
PROMPT_TEMPLATE = """
You are a careful class resolver assistant. You are given a SMALL GROUP of class candidates (each with id, label, description, class_type_hint, evidence_excerpt, and a list of member entities where each member has id, entity_name, entity_description, and entity_type_hint).

Your job: return an ORDERED LIST of ACTIONS (JSON array). Each action must be one of the allowed functions listed below. The ordering matters: your top-to-bottom list will be executed sequentially by the orchestrator.

ALLOWED ACTIONS (JSON objects):
- MergeClasses: {{ "fn":"MergeClasses", "class_ids":[...], "new_name":"...", "new_desc":"...", "new_class_type_hint":"..." }}
- CreateClass: {{ "fn":"CreateClass", "name":"...", "desc":"...", "class_type_hint":"...", "member_ids":[...] }}
- ReassignEntity: {{ "fn":"ReassignEntity", "entity_id":"...", "from_class_id": "...|null", "to_class_id":"..." }}
- ModifyClass: {{ "fn":"ModifyClass", "class_id":"...", "new_name": "...|null", "new_desc": "...|null", "new_class_type_hint":"...|null" }}

RULES:
1) Only use functions above. Do NOT invent other function names.
2) When merging, prefer merging obvious duplicates / near-duplicates. If you propose a merge, always provide a good short new_name and new_desc (1-2 sentences).
3) If you create a class, choose a concise label (1-3 words) and short description and optionally attach member_ids (use provided entity ids).
4) When reassigning entities, include exact entity_id and the from/to class ids from the provided input (if unsure, use null for from_class_id to indicate attach-only).
5) When modifying, provide only fields you want changed.
6) Try to minimize unnecessary changes. Be conservative.

INPUT (classes in this cluster):
{cluster_json}

Return ONLY a JSON array (no markdown). Keep the array small and clearly justified.
"""

def orchestrate_class_resolution():
    # load classes file
    if not CLASSES_ROUND_FILE.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASSES_ROUND_FILE}")
    classes_round = load_json(CLASSES_ROUND_FILE)
    # classes_round expected format: { "n_classes":..., "classes": { "<cid>": {...}, ... } } (as produced earlier)
    raw_classes_map = classes_round.get("classes", {}) if isinstance(classes_round, dict) else {}
    # flatten into list of class dicts with members as list
    classes = []
    for cid, meta in raw_classes_map.items():
        c = dict(meta)
        c["id"] = cid
        # ensure members are full objects or at least id/name/desc/type
        members = c.get("members") or []
        normalized_members = []
        for m in members:
            if isinstance(m, dict) and "id" in m:
                normalized_members.append({
                    "id": m.get("id"),
                    "entity_name": m.get("entity_name",""),
                    "entity_description": m.get("entity_description",""),
                    "entity_type_hint": m.get("entity_type_hint","")
                })
        c["members"] = normalized_members
        classes.append(c)

    # load entity index (to use when we create/reassign)
    entities = {}
    for e in load_jsonl(CLS_INPUT_ENTITIES):
        entities[e.get("id")] = e

    # build embeddings for classes
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_class_embeddings(embedder, classes, weights=WEIGHTS)
    labels, clusterer = run_hdbscan(combined)
    print(f"[cls_res] clustered {len(classes)} classes -> unique labels: {len(set(labels))}")

    # prepare mutable state: class id -> class dict
    state_classes = {c["id"]: c for c in classes}
    canonical_store = []
    log_entries = []

    # group by cluster label (including -1)
    cluster_map = {}
    for idx, lab in enumerate(labels):
        cluster_map.setdefault(int(lab), []).append(idx)

    cluster_ids = sorted([k for k in cluster_map.keys() if k != -1])
    print(f"[cls_res] resolving {len(cluster_ids)} clusters (skipping -1 noise)")

    for cid in cluster_ids:
        idxs = cluster_map[cid]
        cluster_classes = [classes[i] for i in idxs]
        # build compact JSON input for LLM (we pass ALL member objects with id/name/desc/type)
        cluster_payload = []
        for c in cluster_classes:
            cluster_payload.append({
                "id": c.get("id"),
                "class_label": c.get("class_label",""),
                "class_description": c.get("class_description",""),
                "class_type_hint": c.get("class_type_hint",""),
                "evidence_excerpt": c.get("evidence_excerpt",""),
                "members": c.get("members", [])
            })
        cluster_json = json.dumps(cluster_payload, ensure_ascii=False, indent=2)
        prompt = PROMPT_TEMPLATE.format(cluster_json=cluster_json)
        est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
        if est_tokens > MAX_PROMPT_TOKENS_EST:
            # fail safe: if prompt too large, skip and log
            log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"skip_cluster_prompt_too_large", "cluster": cid, "est_tokens": est_tokens})
            print(f"[cls_res] skipped cluster {cid} - prompt too large (est {est_tokens} tokens)")
            continue

        print(f"[cls_res] calling LLM for cluster {cid} (size={len(cluster_classes)}) ...")
        llm_out = call_llm(prompt)
        actions = extract_json_from_text(llm_out)
        if not actions or not isinstance(actions, list):
            # fallback: no actions -> keep classes as-is, log conservative keep
            log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"no_actions_from_llm", "cluster": cid})
            print(f"[cls_res] LLM returned no actionable JSON for cluster {cid}; skipping")
            continue

        # execute actions in order
        for act in actions:
            if not isinstance(act, dict):
                continue
            fn = act.get("fn") or act.get("action") or ""
            fn = fn.strip()
            try:
                if fn == "MergeClasses" or fn.lower()=="mergeclasses":
                    class_ids = act.get("class_ids", []) or []
                    new_name = act.get("new_name") or act.get("name") or ("Merged_" + uuid.uuid4().hex[:6])
                    new_desc = act.get("new_desc") or act.get("description") or ""
                    new_type = act.get("new_class_type_hint") or act.get("new_class_type") or ""
                    can_id = fn_merge_classes(state_classes, class_ids, new_name, new_desc, new_type, canonical_store, log_entries)
                    print(f"[cls_res] MergeClasses -> {can_id} from {class_ids}")
                elif fn == "CreateClass" or fn.lower()=="createclass":
                    name = act.get("name") or act.get("class_label") or ("NewClass_" + uuid.uuid4().hex[:6])
                    desc = act.get("desc") or act.get("description") or ""
                    ctype = act.get("class_type_hint") or ""
                    member_ids = act.get("member_ids", []) or []
                    new_cid = fn_create_class(state_classes, name, desc, ctype, member_ids, entities, log_entries)
                    print(f"[cls_res] CreateClass -> {new_cid} (members={len(member_ids)})")
                elif fn == "ReassignEntity" or fn.lower()=="reassignentity":
                    entity_id = act.get("entity_id")
                    from_c = act.get("from_class_id")
                    to_c = act.get("to_class_id")
                    ok = fn_reassign_entity(state_classes, entity_id, from_c, to_c, entities, log_entries)
                    print(f"[cls_res] ReassignEntity {entity_id} -> {to_c} (ok={ok})")
                elif fn == "ModifyClass" or fn.lower()=="modifyclass":
                    class_id = act.get("class_id")
                    new_name = act.get("new_name")
                    new_desc = act.get("new_desc")
                    new_type = act.get("new_class_type_hint")
                    ok = fn_modify_class(state_classes, class_id, new_name, new_desc, new_type, log_entries)
                    print(f"[cls_res] ModifyClass {class_id} (ok={ok})")
                else:
                    # unknown action -> log and skip
                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"unknown_fn", "payload": act, "cluster": cid})
                    print(f"[cls_res] unknown function requested by LLM: {fn} -> skipped")
            except Exception as e:
                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"fn_exec_error", "fn": fn, "error": str(e)})
                print(f"[cls_res] error executing {fn}: {e}")

    # After cluster loop: collect resolved classes (state_classes)
    final_classes = list(state_classes.values())
    # write outputs
    with open(OUT_CLASSES_JSONL, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(OUT_CANONICAL_JSONL, "w", encoding="utf-8") as fh:
        for can in canonical_store:
            fh.write(json.dumps(can, ensure_ascii=False) + "\n")
    with open(OUT_LOG_JSONL, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    print(f"[cls_res] finished. Wrote: {OUT_CLASSES_JSONL}, {OUT_CANONICAL_JSONL}, {OUT_LOG_JSONL}")

if __name__ == "__main__":
    orchestrate_class_resolution()


#endregion#? Cls Res - Class Resolution
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Cls Res V2



#!/usr/bin/env python3
"""
classres_iterative_v1.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify) for each cluster,
then execute those functions locally and produce final resolved classes.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: cluster_<N>_llm_raw.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings (you can edit)
# fields: class_label, class_desc, class_type_hint, evidence_excerpt, members_agg
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label",""))[:120])
        descs.append(safe_str(c.get("class_description",""))[:300])
        types.append(safe_str(c.get("class_type_hint",""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt",""))[:200])
        # aggregate member short texts
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name",""))
            desc = safe_str(m.get("entity_description",""))
            etype = safe_str(m.get("entity_type_hint",""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str,float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc  = embedder.encode_batch(descs)  if any(t.strip() for t in descs) else None
    emb_type  = embedder.encode_batch(types)  if any(t.strip() for t in types) else None
    emb_evid  = embedder.encode_batch(evids)  if any(t.strip() for t in evids) else None
    emb_mem   = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label); emb_desc = ensure(emb_desc); emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid); emb_mem = ensure(emb_mem)

    w_label = weights.get("label",0.0); w_desc = weights.get("desc",0.0)
    w_type = weights.get("type_hint",0.0); w_evid = weights.get("evidence",0.0)
    w_mem  = weights.get("members",0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0: raise ValueError("invalid class emb weights")
    w_label /= W; w_desc /= W; w_type /= W; w_evid /= W; w_mem /= W

    combined = (w_label*emb_label) + (w_desc*emb_desc) + (w_type*emb_type) + (w_evid*emb_evid) + (w_mem*emb_mem)
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP: require N reasonably larger than target dims/neighbors
    if use_umap and UMAP_AVAILABLE and N >= 6:
        # choose n_components and n_neighbors safely relative to N
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))  # leave small margin
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric='cosine',
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            # Catch UMAP failures (including the scipy eigh TypeError) and continue with original embeddings
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, safe_n_components={safe_n_components}, safe_n_neighbors={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings  # fallback
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    # Run HDBSCAN on X (either reduced or original)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer



# ---------------------- LLM prompt template -----------------------------
# LLM MUST return a JSON array of objects describing ordered function calls.
# Example:
# [
#   {"function": "merge_classes", "args": {"class_ids":["ClsA","ClsB"], "new_name":"X", "new_description":"...", "new_class_type_hint":"Material"}},
#   {"function": "reassign_entities", "args": {"entity_ids":["En_1"], "from_class_id":"ClsA", "to_class_id":"ClsC"}}
# ]
#
# Allowed functions: merge_classes, create_class, reassign_entities, modify_class
CLSRES_PROMPT_TEMPLATE = """
You are a careful class resolver assistant.
Input: a set of candidate classes (with metadata) that appear to belong to the same cluster.
Your job: produce an ordered sequence of function calls (JSON array) that, if executed in order, will best resolve duplicates, ambiguous labels, and mis-assigned members in this cluster.

IMPORTANT CONSERVATISM RULE
- Only order a function if a real change is necessary.
- If a class (or entity assignment) is already correct and coherent, DO NOT order any function for it.
- It is valid and encouraged to return an EMPTY ARRAY [] if no changes are needed for this cluster.
- Do NOT “clean up”, rephrase, or reorganize classes unless there is a concrete semantic problem.


-- Important:
- Return ONLY valid JSON: an array of objects.
- Each object must have:
   - "function": one of ["merge_classes","create_class","reassign_entities","modify_class"]
   - "args": an object with named arguments (see below).

-- Allowed functions and required args:
1) merge_classes: args = {
      "class_ids": [<existing_class_ids>],
      "new_name": <string or null>,
      "new_description": <string or null>,
      "new_class_type_hint": <string or null>
   }
   Semantics: create a single merged class from the union of members.

2) create_class: args = {
      "name": <string>,
      "description": <string or null>,
      "class_type_hint": <string or null>,
      "member_ids": [<entity ids>]  # optional
   }

3) reassign_entities: args = {
      "entity_ids": [<entity ids>],
      "from_class_id": <existing_class_id or null>,
      "to_class_id": <existing_or_new_class_id>
   }

4) modify_class: args = {
      "class_id": <existing_class_id>,
      "new_name": <string or null>,
      "new_description": <string or null>,
      "new_class_type_hint": <string or null>
   }

-- Validation rules I will apply after you respond:
- I will only accept entity ids that are present in the provided member lists.
- I will only accept class_ids that exist in the provided cluster (except 'to_class_id' for create_class results where I'll use the returned new id).
- I will ignore any function objects that are malformed.

-- Strategy notes:
- Prefer merging obviously-duplicate classes (same meaning, small label variation).
- Prefer reassigning mis-assigned members instead of making broad classes.
- If two classes have substantial overlapping members and complementary descriptions, merge them and set an inclusive name.
- Be conservative: if uncertain, create a new narrow class rather than forcing a broad change.

-- Input CLASSES (each includes: candidate_id, class_label, class_description, class_type_hint, confidence, evidence_excerpt, member_ids, members [full member objects]):
{cluster_block}

Return the ordered function list JSON array only.
"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘","'").replace("’","'")
    # find first [ ... ] block
    start = s.find('[')
    end = s.rfind(']')
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        # try eval fallback (risky) -> don't do it. Return None
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(all_classes: Dict[str, Dict], class_ids: List[str], new_name: Optional[str], new_desc: Optional[str], new_type: Optional[str]) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if not class_ids:
        raise ValueError("merge_classes: no valid class_ids provided")
    # create new candidate id
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    # union members
    members_map = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence",0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        for m in c.get("members", []):
            members_map[m["id"]] = m
    # prefer provided new_desc if any
    if new_desc:
        desc_choice = new_desc
    # prefer provided new_name else take first label
    new_label = new_name or all_classes[class_ids[0]].get("class_label","MergedClass")
    if not new_type:
        # attempt to choose highest-confidence type_hint present
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break
    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    # remove old classes and insert new one
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(all_classes: Dict[str, Dict], name: str, description: Optional[str], class_type_hint: Optional[str], member_ids: Optional[List[str]], id_to_entity: Dict[str, Dict]) -> str:
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(all_classes: Dict[str, Dict], entity_ids: List[str], from_class_id: Optional[str], to_class_id: str, id_to_entity: Dict[str, Dict]):
    # If from_class_id is None, we will remove entity from any class that contains it
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        # remove the entity ids from this class
        before = set(c.get("member_ids", []))
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    # deduplicate
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence",0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(all_classes: Dict[str, Dict], class_id: str, new_name: Optional[str], new_desc: Optional[str], new_type: Optional[str]):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    all_classes[class_id] = c

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members contained in classes); optionally load src entity file if needed
    id_to_entity = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        # normalize member list to objects
        members = c.get("members", []) or []
        # ensure member_ids
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end
    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")
        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            # compact members
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label",""),
                "class_description": c.get("class_description",""),
                "class_type_hint": c.get("class_type_hint",""),
                "confidence": float(c.get("confidence",0.0)),
                "evidence_excerpt": c.get("evidence_excerpt",""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact
            })
        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        # prompt = CLSRES_PROMPT_TEMPLATE.format(cluster_block=cluster_block)
        # safer substitution: do not use str.format which treats braces as placeholders
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)


        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            # if cluster is singletons, consider no-op; else log and skip
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            # still write a decision file with raw output
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        # execute parsed function list in order
        decisions = []
        for step in parsed:
            if not isinstance(step, dict): continue
            fn = step.get("function")
            args = step.get("args", {}) or {}
            try:
                if fn == "merge_classes":
                    cids = args.get("class_ids", [])
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    # validate provided class ids: ensure they are in this cluster or globally present
                    valid_cids = [cid for cid in cids if cid in all_classes]
                    if not valid_cids:
                        raise ValueError("no valid class_ids for merge")
                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)
                    decisions.append({"action":"merge_classes","input_class_ids": valid_cids, "result_class_id": new_cid})
                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    # filter mids to known entity ids
                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)
                    decisions.append({"action":"create_class","result_class_id": new_cid, "member_ids_added": mids_valid})
                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c = args.get("from_class_id")
                    to_c = args.get("to_class_id")
                    # ensure eids valid
                    eids_valid = [e for e in eids if e in id_to_entity]
                    # if to_c is not present but matches pattern of created class in prior decisions, allow it
                    if to_c not in all_classes:
                        # check if to_c is one of the new classes created earlier in this cluster run
                        # We support referencing the new class id returned earlier as result_class_id
                        # If not found, error
                        raise ValueError(f"to_class_id {to_c} not found")
                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)
                    decisions.append({"action":"reassign_entities","entity_ids": eids_valid, "from": from_c, "to": to_c})
                elif fn == "modify_class":
                    cid = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    execute_modify_class(all_classes, cid, new_name, new_desc, new_type)
                    decisions.append({"action":"modify_class","class_id": cid, "new_name": new_name})
                else:
                    # unknown function -> skip
                    decisions.append({"action":"skip_unknown","raw": step})
            except Exception as e:
                decisions.append({"action":"error_executing","function": fn, "error": str(e), "input": step})

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

if __name__ == "__main__":
    classres_main()




#endregion#? Cls Res V2
#?#########################  End  ##########################









#?######################### Start ##########################
#region:#?   Cls Res V3 



#!/usr/bin/env python3
"""
classres_iterative_v2.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify) for each cluster,
then execute those functions locally and produce final resolved classes.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: cluster_<N>_llm_raw.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings (you can edit)
# fields: class_label, class_desc, class_type_hint, evidence_excerpt, members_agg
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4.1" # "gpt-5.2-pro" #"gpt-4o"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label",""))[:120])
        descs.append(safe_str(c.get("class_description",""))[:300])
        types.append(safe_str(c.get("class_type_hint",""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt",""))[:200])
        # aggregate member short texts
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name",""))
            desc = safe_str(m.get("entity_description",""))
            etype = safe_str(m.get("entity_type_hint",""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str,float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc  = embedder.encode_batch(descs)  if any(t.strip() for t in descs) else None
    emb_type  = embedder.encode_batch(types)  if any(t.strip() for t in types) else None
    emb_evid  = embedder.encode_batch(evids)  if any(t.strip() for t in evids) else None
    emb_mem   = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label); emb_desc = ensure(emb_desc); emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid); emb_mem = ensure(emb_mem)

    w_label = weights.get("label",0.0); w_desc = weights.get("desc",0.0)
    w_type = weights.get("type_hint",0.0); w_evid = weights.get("evidence",0.0)
    w_mem  = weights.get("members",0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0: raise ValueError("invalid class emb weights")
    w_label /= W; w_desc /= W; w_type /= W; w_evid /= W; w_mem /= W

    combined = (w_label*emb_label) + (w_desc*emb_desc) + (w_type*emb_type) + (w_evid*emb_evid) + (w_mem*emb_mem)
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP: require N reasonably larger than target dims/neighbors
    if use_umap and UMAP_AVAILABLE and N >= 6:
        # choose n_components and n_neighbors safely relative to N
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))  # leave small margin
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric='cosine',
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            # Catch UMAP failures (including the scipy eigh TypeError) and continue with original embeddings
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, safe_n_components={safe_n_components}, safe_n_neighbors={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings  # fallback
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    # Run HDBSCAN on X (either reduced or original)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template (confirmed) -----------------
CLSRES_PROMPT_TEMPLATE = """
You are a careful class resolution assistant.

You are given a set of candidate CLASSES that appear to belong,
or may plausibly belong, to the same semantic cluster.
This grouping is NOT guaranteed to be correct and should be treated as suggestive,
not definitive.

Your task is to conservatively refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

For the given cluster of classes:

1) Assess whether any structural changes are REQUIRED:
   - merge truly duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class ONLY if strictly necessary
   - modify class metadata ONLY if meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct, assigning or confirming Class_Group ALONE is sufficient.

Class_Group assignment is the MINIMUM expected outcome of this step.

========================
IMPORTANT CONSERVATISM RULE
========================

- Only order a function if a real semantic change is necessary.
- Do NOT perform cosmetic edits or unnecessary normalization.
- Do NOT rephrase labels or descriptions unless meaningfully wrong.
- If classes and entity memberships are already correct:
  → DO NOT order merge, create, reassign, or modify actions beyond Class_Group.

BALANCE & ORDERING (SHORT)
- Class_Group is required, but do NOT stop there: if there is clear evidence (overlapping members, contradictory descriptions, high-confidence mismatch) perform merges/reassigns/creates/modifies as needed.
- If you perform any structural changes, do them first; always include a final modify_class that sets or confirms the Class_Group as the last step for the cluster.


========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class"]
- "args": arguments as defined below.

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>]   # optional
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or null>,
  "to_class_id": <existing_or_new_class_id>
}

4) modify_class
args = {
  "class_id": <existing_class_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string>
}


========================
VALIDATION RULES
========================

- Use ONLY provided class_ids and entity_ids.
- Do NOT invent new entity ids or functions.
- Order matters: later steps may depend on earlier ones.

========================
STRATEGY GUIDANCE
========================

- Merge classes ONLY when they are genuinely redundant.
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity, not cleanup.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] ONLY if you are absolutely certain that even Class_Group is already correct.
"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘","'").replace("’","'")
    # find first [ ... ] block
    start = s.find('[')
    end = s.rfind(']')
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        # try eval fallback (risky) -> don't do it. Return None
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(all_classes: Dict[str, Dict], class_ids: List[str], new_name: Optional[str], new_desc: Optional[str], new_type: Optional[str]) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if not class_ids:
        raise ValueError("merge_classes: no valid class_ids provided")
    # create new candidate id
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    # union members
    members_map = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None
    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence",0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        # prefer an existing class_group if all share same - else keep TBD
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep TBD unless new provided
                class_group_choice = class_group_choice  # keep first
        for m in c.get("members", []):
            members_map[m["id"]] = m
    # prefer provided new_desc if any
    if new_desc:
        desc_choice = new_desc
    # prefer provided new_name else take first label
    new_label = new_name or all_classes[class_ids[0]].get("class_label","MergedClass")
    if not new_type:
        # attempt to choose highest-confidence type_hint present
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break
    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    # remove old classes and insert new one
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(all_classes: Dict[str, Dict], name: str, description: Optional[str], class_type_hint: Optional[str], member_ids: Optional[List[str]], id_to_entity: Dict[str, Dict]) -> str:
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(all_classes: Dict[str, Dict], entity_ids: List[str], from_class_id: Optional[str], to_class_id: str, id_to_entity: Dict[str, Dict]):
    # If from_class_id is None, we will remove entity from any class that contains it
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        # remove the entity ids from this class
        before = set(c.get("member_ids", []))
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    # deduplicate
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence",0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(all_classes: Dict[str, Dict], class_id: str, new_name: Optional[str], new_desc: Optional[str], new_type: Optional[str], new_class_group: Optional[str]):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    all_classes[class_id] = c

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members contained in classes); optionally load src entity file if needed
    id_to_entity = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys and class_group field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        # normalize member list to objects
        members = c.get("members", []) or []
        # ensure member_ids
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        # ensure class_group present
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end
    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")
        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            # compact members
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label",""),
                "class_description": c.get("class_description",""),
                "class_type_hint": c.get("class_type_hint",""),
                "class_group": c.get("class_group","TBD"),
                "confidence": float(c.get("confidence",0.0)),
                "evidence_excerpt": c.get("evidence_excerpt",""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact
            })
        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        # safer substitution: do not use str.format which treats braces as placeholders
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            # if cluster is singletons, consider no-op; else log and skip
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            # still write a decision file with raw output
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        # execute parsed function list in order
        decisions = []
        for step in parsed:
            if not isinstance(step, dict): continue
            fn = step.get("function")
            args = step.get("args", {}) or {}
            try:
                if fn == "merge_classes":
                    cids = args.get("class_ids", [])
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    # validate provided class ids: ensure they are in this cluster or globally present
                    valid_cids = [cid for cid in cids if cid in all_classes]
                    if not valid_cids:
                        raise ValueError("no valid class_ids for merge")
                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)
                    decisions.append({"action":"merge_classes","input_class_ids": valid_cids, "result_class_id": new_cid})
                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    # filter mids to known entity ids
                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)
                    decisions.append({"action":"create_class","result_class_id": new_cid, "member_ids_added": mids_valid})
                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c = args.get("from_class_id")
                    to_c = args.get("to_class_id")
                    # ensure eids valid
                    eids_valid = [e for e in eids if e in id_to_entity]
                    # to_c must exist
                    if to_c not in all_classes:
                        raise ValueError(f"to_class_id {to_c} not found")
                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)
                    decisions.append({"action":"reassign_entities","entity_ids": eids_valid, "from": from_c, "to": to_c})
                elif fn == "modify_class":
                    cid = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")
                    execute_modify_class(all_classes, cid, new_name, new_desc, new_type, new_group)
                    decisions.append({"action":"modify_class","class_id": cid, "new_name": new_name, "new_class_group": new_group})
                else:
                    # unknown function -> skip
                    decisions.append({"action":"skip_unknown","raw": step})
            except Exception as e:
                decisions.append({"action":"error_executing","function": fn, "error": str(e), "input": step})

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

if __name__ == "__main__":
    classres_main()



#endregion#? Cls Res V3
#?#########################  End  ##########################














#?######################### Start ##########################
#region:#?   Cls Res V4 

#!/usr/bin/env python3
"""
classres_iterative_v4.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify) for each cluster,
then execute those functions locally and produce final resolved classes.

Key features:
- TWO-LAYER SCHEMA: Class_Group -> Classes -> Entities
- class_label treated as provisional; may be revised if evidence suggests a clearer name.
- LLM orders structural + schema actions using a small function vocabulary.
- Provisional IDs for newly created/merged classes so later steps can refer to them.
- Conservative behavior with strong validation and logging.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings (you can edit)
# fields: class_label, class_desc, class_type_hint, evidence_excerpt, members_agg
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4.1"  # adjust as needed
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 3000
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label",""))[:120])
        descs.append(safe_str(c.get("class_description",""))[:300])
        types.append(safe_str(c.get("class_type_hint",""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt",""))[:200])
        # aggregate member short texts
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name",""))
            desc = safe_str(m.get("entity_description",""))
            etype = safe_str(m.get("entity_type_hint",""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str,float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc  = embedder.encode_batch(descs)  if any(t.strip() for t in descs) else None
    emb_type  = embedder.encode_batch(types)  if any(t.strip() for t in types) else None
    emb_evid  = embedder.encode_batch(evids)  if any(t.strip() for t in evids) else None
    emb_mem   = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label); emb_desc = ensure(emb_desc); emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid); emb_mem = ensure(emb_mem)

    w_label = weights.get("label",0.0); w_desc = weights.get("desc",0.0)
    w_type = weights.get("type_hint",0.0); w_evid = weights.get("evidence",0.0)
    w_mem  = weights.get("members",0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0: raise ValueError("invalid class emb weights")
    w_label /= W; w_desc /= W; w_type /= W; w_evid /= W; w_mem /= W

    combined = (w_label*emb_label) + (w_desc*emb_desc) + (w_type*emb_type) + (w_evid*emb_evid) + (w_mem*emb_mem)
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP: require N reasonably larger than target dims/neighbors
    if use_umap and UMAP_AVAILABLE and N >= 6:
        # choose n_components and n_neighbors safely relative to N
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))  # leave small margin
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric='cosine',
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            # Catch UMAP failures (including the scipy eigh TypeError) and continue with original embeddings
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, safe_n_components={safe_n_components}, safe_n_neighbors={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings  # fallback
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    # Run HDBSCAN on X (either reduced or original)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template (confirmed + extended) ------
CLSRES_PROMPT_TEMPLATE = """
You are a careful class resolution assistant.
You are given a set of candidate CLASSES that appear to belong,
or may plausibly belong, to the same semantic cluster.

The cluster grouping you are given is only *suggestive* and may be incorrect — your job is to resolve, correct, and produce a coherent schema from the evidence.

This is an iterative process — act now with well-justified structural corrections (include a short justification), rather than deferring small but meaningful fixes.

Your CRUCIAL task is to refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- class_label:
  Existing class_label values are PROVISIONAL names.
  You MAY revise them when entity evidence suggests a clearer or a better canonical label.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

Note: the provided cluster grouping is tentative and may be wrong — 
you must correct it as needed to produce a coherent Class_Group → Class → Entity schema.


For the given cluster of classes:


1) Assess whether any structural changes are REQUIRED:
   - merge duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class when necessary
   - modify class metadata when meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct (which is not the case most of the time), assigning or confirming Class_Group ALONE is sufficient.

========================
SOME CONSERVATISM RULES (They should not make you passive)
========================

- Only order a function if a real semantic change is necessary.
- Do NOT perform cosmetic edits or unnecessary normalization.

You MAY perform multiple structural actions in one cluster
(e.g., merge + rename + reassign), but when needed.

========================
MERGING & OVERLAP HEURISTICS
========================

- Entity overlap alone does not automatically require merging.
- Merge when evidence indicates the SAME underlying concepts
  (e.g., near-identical semantics, interchangeable usage, or redundant distinctions).
- Reassignment is appropriate when overlap reveals mis-typed or mis-scoped entities,
  even if classes should remain separate.


Quick heuristic:
- Same concept → merge.
- Different concept, same domain → keep separate classes with the SAME Class_Group.
- Different domain → different Class_Group.

Avoid vague Class_Group names (e.g., "Misc", "General", "Other").
Prefer domain-meaningful groupings that help connect related classes.

========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class"]
- "args": arguments as defined below.

ID HANDLING RULES
- You MUST NOT invent real class IDs.
- You MUST use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except when referring to newly merged or created classes.
- When you need to refer to a newly merged or created class in later steps,
  you MUST assign a provisional_id (any consistent string).
- Use the same provisional_id whenever referencing that new class again.

Example (pattern, not required verbatim):

{
  "function": "merge_classes",
  "args": {
    "class_ids": ["ClsC_da991b68", "ClsC_e32f4a47"],
    "provisional_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "new_name": "...",
    "new_description": "...",
    "new_class_type_hint": "Standard"
  }
}

Later:

{
  "function": "reassign_entities",
  "args": {
    "entity_ids": ["En_xxx"],
    "from_class_id": "ClsC_e32f4a47",
    "to_class_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)"
  }
}

We will internally map provisional_id → real class id.

------------------------
Function definitions
------------------------

JUSTIFICATION REQUIREMENT
- Every function call MUST include a one-line "justification" explaining why the action is necessary.
- This justification should cite concrete evidence (entity overlap, conflicting descriptions, mis-scoped members, etc.).
- The justification is required for ALL actions, including merge, create, reassign, and modify.


1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,    # how you will refer to the new class later
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>]   # optional, must be from provided entities
  "provisional_id": <string or null>   # how you will refer to this new class later
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>
}

NOTE:
- Class_Group is normally set or updated via modify_class.
- Assigning or confirming Class_Group is also REQUIRED for every cluster unless it is already clearly correct.
- Optionally include "confidence": <0.0-1.0> and "justification": "<one-line reason>" in each function's args to help downstream acceptance.


========================
VALIDATION RULES
========================

- Use ONLY provided entity_ids and class_ids (candidate_id values) for existing classes.
- For new classes, use provisional_id handles and be consistent.
- Order matters: later steps may depend on earlier ones.
- merge_classes with fewer than 2 valid class_ids will be ignored.

========================
STRATEGY GUIDANCE
========================

- Prefer assigning/adjusting Class_Group over heavy structural changes.
- Merge classes ONLY when they are genuinely redundant (same concept).
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity, not cosmetic cleanup.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] only if you are highly confident (>very strong evidence) that no change is needed. 
If any ambiguous or conflicting evidence exists, return a concrete ordered action list (you may include low-confidence recommendations with brief justification).
"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘","'").replace("’","'")
    # find first [ ... ] block
    start = s.find('[')
    end = s.rfind(']')
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(
    all_classes: Dict[str, Dict],
    class_ids: List[str],
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str]
) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if len(class_ids) < 2:
        raise ValueError("merge_classes: need at least 2 valid class_ids")
    # create new candidate id
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    # union members
    members_map = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None

    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence", 0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        # prefer an existing class_group if all share same - else keep TBD
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep the first, will be fixable by modify_class
                class_group_choice = class_group_choice
        for m in c.get("members", []):
            members_map[m["id"]] = m

    # prefer provided new_desc if any
    if new_desc:
        desc_choice = new_desc
    # prefer provided new_name else take first label
    new_label = new_name or all_classes[class_ids[0]].get("class_label", "MergedClass")
    if not new_type:
        # attempt to choose highest-confidence type_hint present
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break

    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    # remove old classes and insert new one
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(
    all_classes: Dict[str, Dict],
    name: str,
    description: Optional[str],
    class_type_hint: Optional[str],
    member_ids: Optional[List[str]],
    id_to_entity: Dict[str, Dict]
) -> str:
    if not name:
        raise ValueError("create_class: 'name' is required")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict]
):
    # If from_class_id is None, we will remove entity from any class that contains it
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    # deduplicate
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence", 0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(
    all_classes: Dict[str, Dict],
    class_id: str,
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str],
    new_class_group: Optional[str]
):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    all_classes[class_id] = c

# helper to resolve real class ID (handles provisional IDs)
def resolve_class_id(
    raw_id: Optional[str],
    all_classes: Dict[str, Dict],
    provisional_to_real: Dict[str, str],
    allow_missing: bool = False
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes")
    return real

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members contained in classes); optionally load src entity file if needed
    id_to_entity = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys and class_group field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        # normalize member list to objects
        members = c.get("members", []) or []
        # ensure member_ids
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        # ensure class_group present
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end

    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")

        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label", ""),
                "class_description": c.get("class_description", ""),
                "class_type_hint": c.get("class_type_hint", ""),
                "class_group": c.get("class_group", "TBD"),
                "confidence": float(c.get("confidence", 0.0)),
                "evidence_excerpt": c.get("evidence_excerpt", ""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact
            })

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        # safer substitution: do not use str.format which treats braces as placeholders
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            continue

        # mapping from provisional_id -> real class id (per cluster)
        provisional_to_real: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}
            try:
                if fn == "merge_classes":
                    cids_raw = args.get("class_ids", []) or []
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    prov_id = args.get("provisional_id")

                    # resolve any potential provisional IDs in class_ids (ideally they should be real)
                    cids_real = [provisional_to_real.get(cid, cid) for cid in cids_raw]
                    valid_cids = [cid for cid in cids_real if cid in all_classes]

                    if len(valid_cids) < 2:
                        # enforce executor rule: skip if < 2
                        decisions.append({
                            "action": "merge_skip_too_few",
                            "requested_class_ids": cids_raw,
                            "valid_class_ids": valid_cids
                        })
                        continue

                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "merge_classes",
                        "input_class_ids": valid_cids,
                        "result_class_id": new_cid,
                        "provisional_id": prov_id
                    })

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    # filter mids to known entity ids
                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "create_class",
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "member_ids_added": mids_valid
                    })

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    # ensure eids valid
                    eids_valid = [e for e in eids if e in id_to_entity]

                    # resolve class ids (handle provisional ids)
                    from_c = resolve_class_id(from_c_raw, all_classes, provisional_to_real, allow_missing=True)
                    to_c = resolve_class_id(to_c_raw, all_classes, provisional_to_real, allow_missing=False)

                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)
                    decisions.append({
                        "action": "reassign_entities",
                        "entity_ids": eids_valid,
                        "from": from_c_raw,
                        "from_resolved": from_c,
                        "to": to_c_raw,
                        "to_resolved": to_c
                    })

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")

                    cid_real = resolve_class_id(cid_raw, all_classes, provisional_to_real, allow_missing=False)
                    execute_modify_class(all_classes, cid_real, new_name, new_desc, new_type, new_group)

                    decisions.append({
                        "action": "modify_class",
                        "class_id": cid_raw,
                        "class_id_resolved": cid_real,
                        "new_name": new_name,
                        "new_class_group": new_group
                    })

                else:
                    # unknown function -> skip
                    decisions.append({"action": "skip_unknown_function", "raw": step})

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step
                })

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

if __name__ == "__main__":
    classres_main()

#endregion#? Cls Res V4
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Cls Res V5  

#!/usr/bin/env python3
"""
classres_iterative_v5.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify) for each cluster,
then execute those functions locally and produce final resolved classes.

Key features:
- TWO-LAYER SCHEMA: Class_Group -> Classes -> Entities
- class_label treated as provisional; may be revised if evidence suggests a clearer name.
- LLM orders structural + schema actions using a small function vocabulary.
- Provisional IDs for newly created/merged classes so later steps can refer to them.
- Conservative behavior with strong validation and logging.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4.1"  # adjust as needed
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 3000
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE:
            print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label", ""))[:120])
        descs.append(safe_str(c.get("class_description", ""))[:300])
        types.append(safe_str(c.get("class_type_hint", ""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt", ""))[:200])
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name", ""))
            desc = safe_str(m.get("entity_description", ""))
            etype = safe_str(m.get("entity_type_hint", ""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str, float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    emb_evid = embedder.encode_batch(evids) if any(t.strip() for t in evids) else None
    emb_mem = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label)
    emb_desc = ensure(emb_desc)
    emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid)
    emb_mem = ensure(emb_mem)

    w_label = weights.get("label", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_type = weights.get("type_hint", 0.0)
    w_evid = weights.get("evidence", 0.0)
    w_mem = weights.get("members", 0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0:
        raise ValueError("invalid class emb weights")
    w_label /= W
    w_desc /= W
    w_type /= W
    w_evid /= W
    w_mem /= W

    combined = (
        w_label * emb_label
        + w_desc * emb_desc
        + w_type * emb_type
        + w_evid * emb_evid
        + w_mem * emb_mem
    )
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP
) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP
    if use_umap and UMAP_AVAILABLE and N >= 6:
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, n_comp={safe_n_components}, n_nei={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template (confirmed + extended) ------
CLSRES_PROMPT_TEMPLATE = """
You are a careful class resolution assistant.
You are given a set of candidate CLASSES that appear to belong,
or may plausibly belong, to the same semantic cluster.

The cluster grouping you are given is only *suggestive* and may be incorrect — your job is to resolve, correct, and produce a coherent schema from the evidence.

This is an iterative process — act now with well-justified structural corrections (include a short justification), rather than deferring small but meaningful fixes.

Your CRUCIAL task is to refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- class_label:
  Existing class_label values are PROVISIONAL names.
  You MAY revise them when entity evidence suggests a clearer or a better canonical label.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

Note: the provided cluster grouping is tentative and may be wrong — 
you must correct it as needed to produce a coherent Class_Group → Class → Entity schema.

For the given cluster of classes:

1) Assess whether any structural changes are REQUIRED:
   - merge duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class when necessary
   - modify class metadata when meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct (which is not the case most of the time), assigning or confirming Class_Group ALONE is sufficient.

========================
SOME CONSERVATISM RULES (They should not make you passive)
========================

- Only order a function if a real semantic change is necessary.
- Do NOT perform cosmetic edits or unnecessary normalization.

You MAY perform multiple structural actions in one cluster
(e.g., merge + rename + reassign), when needed.

========================
MERGING & OVERLAP HEURISTICS
========================

- Entity overlap alone does not automatically require merging.
- Merge when evidence indicates the SAME underlying concepts
  (e.g., near-identical semantics, interchangeable usage, or redundant distinctions).
- Reassignment is appropriate when overlap reveals mis-typed or mis-scoped entities,
  even if classes should remain separate.

Quick heuristic:
- Same concept → merge.
- Different concept, same domain → keep separate classes with the SAME Class_Group.
- Different domain → different Class_Group.

Avoid vague Class_Group names (e.g., "Misc", "General", "Other").
Prefer domain-meaningful groupings that help connect related classes.

If a class should be collapsed or weakened but has no clear merge partner, DO NOT use merge_classes;
instead, reassign its entities to better classes and leave the class unchanged or empty.

IMPORTANT:
- You MUST NOT call merge_classes with only one class_id.
- If you only want to update or clarify a single class (e.g., it already contains redundant entities),
  use modify_class instead, and leave class_ids out of merge_classes.
- Do NOT use merge_classes to clean up or deduplicate entities inside one class.


========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class"]
- "args": arguments as defined below.

ID HANDLING RULES
- You MUST NOT invent real class IDs.
- You MUST use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except when referring to newly merged or created classes.
- When you need to refer to a newly merged or created class in later steps,
  you MUST assign a provisional_id (any consistent string).
- Use the same provisional_id whenever referencing that new class again.
- After you merge classes into a new class, you should NOT continue to treat the original
  class_ids as separate entities; refer to the new merged class via its provisional_id.

Example (pattern, not required verbatim):

{
  "function": "merge_classes",
  "args": {
    "class_ids": ["ClsC_da991b68", "ClsC_e32f4a47"],
    "provisional_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "new_name": "...",
    "new_description": "...",
    "new_class_type_hint": "Standard",
    "justification": "One-line reason citing entity overlap and semantic equivalence.",
    "confidence": 0.95
  }
}

Later:

{
  "function": "reassign_entities",
  "args": {
    "entity_ids": ["En_xxx"],
    "from_class_id": "ClsC_e32f4a47",
    "to_class_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "justification": "Why this entity fits better in the merged class.",
    "confidence": 0.9
  }
}

We will internally map provisional_id → real class id.

------------------------
Function definitions
------------------------

JUSTIFICATION REQUIREMENT
- Every function call MUST include:
    "justification": "<one-line reason>"
  explaining why the action is necessary.
- This justification should cite concrete evidence (entity overlap, conflicting descriptions, mis-scoped members, etc.).
- You MAY also include "confidence": <0.0–1.0> to indicate your belief in the action.

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,    # how you will refer to the new class later
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>],          # optional, must be from provided entities
  "provisional_id": <string or null>,    # how you will refer to this new class later
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>,
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>,
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

NOTE:
- Class_Group is normally set or updated via modify_class.
- Assigning or confirming Class_Group is also REQUIRED for every cluster unless it is already clearly correct.

========================
VALIDATION RULES
========================

- Use ONLY provided entity_ids and class_ids (candidate_id values) for existing classes.
- For new classes, use provisional_id handles and be consistent.
- Order matters: later steps may depend on earlier ones.
- merge_classes with fewer than 2 valid class_ids will be ignored.

========================
STRATEGY GUIDANCE
========================

- Prefer assigning/adjusting Class_Group over heavy structural changes.
- Merge classes ONLY when they are genuinely redundant (same concept).
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity, not cosmetic cleanup.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] only if you are highly confident (> very strong evidence) that no change is needed.
If any ambiguous or conflicting evidence exists, return a concrete ordered action list
(with justifications and, optionally, confidence scores).
"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # find first [ ... ] block
    start = s.find("[")
    end = s.rfind("]")
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end + 1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(
    all_classes: Dict[str, Dict],
    class_ids: List[str],
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str]
) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if len(class_ids) < 2:
        raise ValueError("merge_classes: need at least 2 valid class_ids")
    # create new candidate id
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    # union members
    members_map: Dict[str, Dict] = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None

    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence", 0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep the first, fixable later by modify_class
                class_group_choice = class_group_choice
        for m in c.get("members", []):
            members_map[m["id"]] = m

    if new_desc:
        desc_choice = new_desc
    new_label = new_name or all_classes[class_ids[0]].get("class_label", "MergedClass")
    if not new_type:
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break

    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(
    all_classes: Dict[str, Dict],
    name: str,
    description: Optional[str],
    class_type_hint: Optional[str],
    member_ids: Optional[List[str]],
    id_to_entity: Dict[str, Dict]
) -> str:
    if not name:
        raise ValueError("create_class: 'name' is required")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict]
):
    # remove from source(s)
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence", 0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(
    all_classes: Dict[str, Dict],
    class_id: str,
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str],
    new_class_group: Optional[str]
):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    all_classes[class_id] = c

# helper to resolve real class ID (handles provisional IDs and retired IDs)
def resolve_class_id(
    raw_id: Optional[str],
    all_classes: Dict[str, Dict],
    provisional_to_real: Dict[str, str],
    allow_missing: bool = False
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes")
    return real

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members)
    id_to_entity: Dict[str, Dict] = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys and class_group field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        members = c.get("members", []) or []
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        c["candidate_id"] = cid
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end

    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")

        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label", ""),
                "class_description": c.get("class_description", ""),
                "class_type_hint": c.get("class_type_hint", ""),
                "class_group": c.get("class_group", "TBD"),
                "confidence": float(c.get("confidence", 0.0)),
                "evidence_excerpt": c.get("evidence_excerpt", ""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact
            })

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            continue

        # mapping from provisional_id (and retired ids) -> real class id (per cluster)
        provisional_to_real: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}

            justification = args.get("justification")
            confidence_val = args.get("confidence", None)

            try:
                if fn == "merge_classes":
                    cids_raw = args.get("class_ids", []) or []
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    prov_id = args.get("provisional_id")

                    # resolve any potential provisional IDs in class_ids
                    cids_real = [provisional_to_real.get(cid, cid) for cid in cids_raw]
                    valid_cids = [cid for cid in cids_real if cid in all_classes]

                    if len(valid_cids) < 2:
                        decisions.append({
                            "action": "merge_skip_too_few",
                            "requested_class_ids": cids_raw,
                            "valid_class_ids": valid_cids,
                            "justification": justification,
                            "confidence": confidence_val
                        })
                        continue

                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)

                    # map provisional id -> new class id
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid
                    # also map old real ids -> new id so later references can still resolve
                    for old in valid_cids:
                        provisional_to_real.setdefault(old, new_cid)

                    decisions.append({
                        "action": "merge_classes",
                        "input_class_ids": valid_cids,
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)

                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "create_class",
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "member_ids_added": mids_valid,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    eids_valid = [e for e in eids if e in id_to_entity]

                    from_c = resolve_class_id(from_c_raw, all_classes, provisional_to_real, allow_missing=True)
                    to_c = resolve_class_id(to_c_raw, all_classes, provisional_to_real, allow_missing=False)

                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)

                    decisions.append({
                        "action": "reassign_entities",
                        "entity_ids": eids_valid,
                        "from": from_c_raw,
                        "from_resolved": from_c,
                        "to": to_c_raw,
                        "to_resolved": to_c,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")

                    cid_real = resolve_class_id(cid_raw, all_classes, provisional_to_real, allow_missing=False)
                    execute_modify_class(all_classes, cid_real, new_name, new_desc, new_type, new_group)

                    decisions.append({
                        "action": "modify_class",
                        "class_id": cid_raw,
                        "class_id_resolved": cid_real,
                        "new_name": new_name,
                        "new_description": new_desc,
                        "new_class_type_hint": new_type,
                        "new_class_group": new_group,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                else:
                    decisions.append({
                        "action": "skip_unknown_function",
                        "raw": step,
                        "justification": justification,
                        "confidence": confidence_val
                    })

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step,
                    "justification": justification,
                    "confidence": confidence_val
                })

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

if __name__ == "__main__":
    classres_main()

#endregion#? Cls Res V5
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Cls Res V6  - Split added - Remark for more expressevity

#!/usr/bin/env python3
"""
classres_iterative_v6.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify/split) for each cluster,
then execute those functions locally and produce final resolved classes.

Key features:
- TWO-LAYER SCHEMA: Class_Group -> Classes -> Entities
- class_label treated as provisional; may be revised if evidence suggests a clearer name.
- LLM orders structural + schema actions using a small function vocabulary.
- Provisional IDs for newly created/merged/split classes so later steps can refer to them.
- 'remarks' channel so the LLM can flag out-of-scope or higher-level concerns without
  misusing structural functions.
- Conservative behavior with strong validation and logging.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4.1"  # adjust as needed
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 3000
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE:
            print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label", ""))[:120])
        descs.append(safe_str(c.get("class_description", ""))[:300])
        types.append(safe_str(c.get("class_type_hint", ""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt", ""))[:200])
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name", ""))
            desc = safe_str(m.get("entity_description", ""))
            etype = safe_str(m.get("entity_type_hint", ""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str, float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    emb_evid = embedder.encode_batch(evids) if any(t.strip() for t in evids) else None
    emb_mem = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label)
    emb_desc = ensure(emb_desc)
    emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid)
    emb_mem = ensure(emb_mem)

    w_label = weights.get("label", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_type = weights.get("type_hint", 0.0)
    w_evid = weights.get("evidence", 0.0)
    w_mem = weights.get("members", 0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0:
        raise ValueError("invalid class emb weights")
    w_label /= W
    w_desc /= W
    w_type /= W
    w_evid /= W
    w_mem /= W

    combined = (
        w_label * emb_label
        + w_desc * emb_desc
        + w_type * emb_type
        + w_evid * emb_evid
        + w_mem * emb_mem
    )
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP
) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP
    if use_umap and UMAP_AVAILABLE and N >= 6:
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, n_comp={safe_n_components}, n_nei={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template (confirmed + extended) ------
CLSRES_PROMPT_TEMPLATE = """
You are a proactive class-resolution assistant.
You are given a set of candidate CLASSES that appear to belong,or may plausibly belong, to the same semantic cluster.
Your job is to produce a clear, *actionable* ordered list of schema edits for the given cluster.  
Do NOT be passive.  If the evidence supports structural change (merge / split / reassign / create), you MUST propose it.  

Only return [] if there is extremely strong evidence that NO change is needed (rare).


The cluster grouping you are given is only *suggestive* and may be incorrect — your job is to resolve, correct, and produce a coherent schema from the evidence.

This is an iterative process — act now with well-justified structural corrections (include a short justification), rather than deferring small but meaningful fixes.

Your CRUCIAL task is to refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- class_label:
  Existing class_label values are PROVISIONAL names.
  You MAY revise them when entity evidence suggests a clearer or a better canonical label.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

- remarks (optional, internal):
  Free-text notes attached to a class, used to flag important issues that are
  outside the scope of structural changes (e.g., suspected entity-level duplicates
  that should be checked elsewhere).

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

Note: the provided cluster grouping is tentative and may be wrong — 
you must correct it as needed to produce a coherent Class_Group → Class → Entity schema.

For the given cluster of classes:

1) Assess whether any structural changes are REQUIRED:
   - merge duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class when necessary
   - split an overloaded class into more coherent subclasses when justified
   - modify class metadata when meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct (which is not the case most of the time), assigning or confirming Class_Group ALONE is sufficient.

3) If you notice important issues that are OUTSIDE the scope of these functions
   (e.g., upstream entity resolution that you suspect is wrong, or entities that
   look identical but should not be changed here), DO NOT try to fix them via merge or reassign.
   Instead, attach a human-facing remark via modify_class (using the 'remark' field)
   so that a human can review it later.

========================
SOME CONSERVATISM RULES (They should not make you passive)
========================


- Always try to attempt structural proposals (merge/split/reassign/create) unless the cluster truly is already optimal.
- Do NOT perform cosmetic edits or unnecessary normalization.

You MAY perform multiple structural actions in one cluster
(e.g., merge + rename + reassign + split), when needed.



========================
MERGING & OVERLAP HEURISTICS
========================

- Entity overlap alone does not automatically require merging.
- Merge when evidence indicates the SAME underlying concepts
  (e.g., near-identical semantics, interchangeable usage, or redundant distinctions).
- Reassignment is appropriate when overlap reveals mis-typed or mis-scoped entities,
  even if classes should remain separate.

Quick heuristic:
- Same concept → merge.
- Different concept, same domain → keep separate classes with the SAME Class_Group.
- Different domain → different Class_Group.

Avoid vague Class_Group names (e.g., "Misc", "General", "Other").
Prefer domain-meaningful groupings that help connect related classes.

If a class should be collapsed or weakened but has no clear merge partner, DO NOT use merge_classes;
instead, reassign its entities to better classes and/or add a remark for human review.

IMPORTANT:
- You MUST NOT call merge_classes with only one class_id.
- merge_classes is ONLY for merging TWO OR MORE existing classes into ONE new class.
- If you only want to update or clarify a single class (for example, its label, description,
  type hint, class_group, or remarks), use modify_class instead.
- Do NOT use merge_classes to clean up or deduplicate entities inside one class.

========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class", "split_class"]
- "args": arguments as defined below.

ID HANDLING RULES
- You MUST NOT invent real class IDs.
- You MUST use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except when referring to newly merged/created/split classes.
- When you need to refer to a newly merged/created/split class in later steps,
  you MUST assign a provisional_id (any consistent string).
- Use the same provisional_id whenever referencing that new class again.
- After you merge classes into a new class, you should NOT continue to treat the original
  class_ids as separate entities; refer to the new merged class via its provisional_id.

Example (pattern, not required verbatim):

{
  "function": "merge_classes",
  "args": {
    "class_ids": ["ClsC_da991b68", "ClsC_e32f4a47"],
    "provisional_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "new_name": "...",
    "new_description": "...",
    "new_class_type_hint": "Standard",
    "justification": "One-line reason citing entity overlap and semantic equivalence.",
    "confidence": 0.95
  }
}

Later:

{
  "function": "reassign_entities",
  "args": {
    "entity_ids": ["En_xxx"],
    "from_class_id": "ClsC_e32f4a47",
    "to_class_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "justification": "Why this entity fits better in the merged class.",
    "confidence": 0.9
  }
}

We will internally map provisional_id → real class id.

------------------------
Function definitions
------------------------

JUSTIFICATION REQUIREMENT
- Every function call MUST include:
    "justification": "<one-line reason>"
  explaining why the action is necessary.
- This justification should cite concrete evidence (entity overlap, conflicting descriptions,
  mis-scoped members, missing Class_Group, overloaded classes, etc.).
- You MAY also include "confidence": <0.0–1.0> to indicate your belief in the action.

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,    # how you will refer to the new class later
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>],          # optional, must be from provided entities
  "provisional_id": <string or null>,    # how you will refer to this new class later
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

- Use modify_class with 'remark' (and no structural change) when you want to flag
  issues that are outside the scope of this step (e.g., suspected entity-level duplicates).

5) split_class
args = {
  "source_class_id": <existing_class_id or provisional_id>,
  "splits": [
    {
      "name": <string or null>,
      "description": <string or null>,
      "class_type_hint": <string or null>,
      "member_ids": [<entity_ids>],      # must be from source_class member_ids
      "provisional_id": <string or null> # how you will refer to this new split class
    },
    ...
  ],
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag

  "confidence": <number between 0 and 1, optional>
}

Semantics of split_class:
- Use split_class when a single class is overloaded and should be divided into
  narrower, more coherent classes.
- Entities listed in splits[*].member_ids MUST come from the source_class's member_ids.
- The specified entities are REMOVED from the source_class and grouped into new classes.
- Any members not mentioned in any split remain in the source_class.

NOTE:
- Class_Group is normally set or updated via modify_class.
- Assigning or confirming Class_Group is also REQUIRED for every cluster unless it is already clearly correct.

========================
VALIDATION RULES
========================

- Use ONLY provided entity_ids and class_ids (candidate_id values) for existing classes.
- For new classes, use provisional_id handles and be consistent.
- Order matters: later steps may depend on earlier ones.
- merge_classes with fewer than 2 valid class_ids will be ignored.

========================
STRATEGY GUIDANCE
========================

- Prefer assigning/adjusting Class_Group over heavy structural changes when both are valid.
- Merge classes ONLY when they are genuinely redundant (same concept).
- Use split_class when a class clearly bundles multiple distinct concepts that should be separated.
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity and meaningful structure, not cosmetic cleanup.
- If in doubt about structural change but you see a potential issue, use modify_class with a 'remark'
  rather than forcing an uncertain structural edit.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)
- remarks (optional list of prior remarks)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] only if you are highly confident (> very strong evidence) that no change is needed.
If any ambiguous or conflicting evidence exists, return a concrete ordered action list
(with justifications and, optionally, confidence scores).
"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # find first [ ... ] block
    start = s.find("[")
    end = s.rfind("]")
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end + 1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(
    all_classes: Dict[str, Dict],
    class_ids: List[str],
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str]
) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if len(class_ids) < 2:
        raise ValueError("merge_classes: need at least 2 valid class_ids")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members_map: Dict[str, Dict] = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None

    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence", 0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep first; fixable later by modify_class
                class_group_choice = class_group_choice
        for m in c.get("members", []):
            members_map[m["id"]] = m

    if new_desc:
        desc_choice = new_desc
    new_label = new_name or all_classes[class_ids[0]].get("class_label", "MergedClass")
    if not new_type:
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break

    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "remarks": [],
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(
    all_classes: Dict[str, Dict],
    name: str,
    description: Optional[str],
    class_type_hint: Optional[str],
    member_ids: Optional[List[str]],
    id_to_entity: Dict[str, Dict]
) -> str:
    if not name:
        raise ValueError("create_class: 'name' is required")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "remarks": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict]
):
    # remove from source(s)
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence", 0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(
    all_classes: Dict[str, Dict],
    class_id: str,
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str],
    new_class_group: Optional[str],
    new_remark: Optional[str]
):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    if new_remark:
        existing = c.get("remarks")
        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [str(existing)]
        existing.append(str(new_remark))
        c["remarks"] = existing
    all_classes[class_id] = c

def execute_split_class(
    all_classes: Dict[str, Dict],
    source_class_id: str,
    splits_specs: List[Dict[str, Any]]
) -> List[Tuple[str, Optional[str], List[str]]]:
    """
    Split a source class into several new classes.
    Returns list of (new_class_id, provisional_id, member_ids_used) for each created class.
    """
    if source_class_id not in all_classes:
        raise ValueError(f"split_class: source_class_id {source_class_id} not found")
    src = all_classes[source_class_id]
    src_members = src.get("members", []) or []
    src_member_map = {m["id"]: m for m in src_members if isinstance(m, dict) and m.get("id")}
    used_ids: set = set()
    created: List[Tuple[str, Optional[str], List[str]]] = []

    for spec in splits_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        desc = spec.get("description")
        th = spec.get("class_type_hint") or src.get("class_type_hint", "")
        mids_raw = spec.get("member_ids", []) or []
        prov_id = spec.get("provisional_id")
        # use only members that belong to source and are not already used
        valid_mids = []
        for mid in mids_raw:
            if mid in src_member_map and mid not in used_ids:
                valid_mids.append(mid)
        if not valid_mids:
            continue
        members = [src_member_map[mid] for mid in valid_mids]
        new_cid = "ClsR_" + uuid.uuid4().hex[:8]
        obj = {
            "candidate_id": new_cid,
            "class_label": name or src.get("class_label", "SplitClass"),
            "class_description": desc or src.get("class_description", ""),
            "class_type_hint": th or src.get("class_type_hint", ""),
            "class_group": src.get("class_group", "TBD"),
            "confidence": src.get("confidence", 0.5),
            "evidence_excerpt": src.get("evidence_excerpt", ""),
            "member_ids": valid_mids,
            "members": members,
            "candidate_ids": [],
            "remarks": [],
            "_split_from": source_class_id,
            "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        all_classes[new_cid] = obj
        used_ids.update(valid_mids)
        created.append((new_cid, prov_id, valid_mids))

    if used_ids:
        remaining_members = [m for m in src_members if m["id"] not in used_ids]
        src["members"] = remaining_members
        src["member_ids"] = [m["id"] for m in remaining_members]
        all_classes[source_class_id] = src

    return created

# helper to resolve real class ID (handles provisional IDs and retired IDs)
def resolve_class_id(
    raw_id: Optional[str],
    all_classes: Dict[str, Dict],
    provisional_to_real: Dict[str, str],
    allow_missing: bool = False
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes")
    return real

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members)
    id_to_entity: Dict[str, Dict] = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys, class_group field, and remarks field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        members = c.get("members", []) or []
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        # normalize remarks
        if "remarks" not in c or c["remarks"] is None:
            c["remarks"] = []
        elif not isinstance(c["remarks"], list):
            c["remarks"] = [str(c["remarks"])]
        c["candidate_id"] = cid
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end

    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")

        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label", ""),
                "class_description": c.get("class_description", ""),
                "class_type_hint": c.get("class_type_hint", ""),
                "class_group": c.get("class_group", "TBD"),
                "confidence": float(c.get("confidence", 0.0)),
                "evidence_excerpt": c.get("evidence_excerpt", ""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact,
                "remarks": c.get("remarks", [])
            })

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            continue

        # mapping from provisional_id (and retired ids) -> real class id (per cluster)
        provisional_to_real: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}

            justification = args.get("justification")
            confidence_val = args.get("confidence", None)

            try:
                if fn == "merge_classes":
                    cids_raw = args.get("class_ids", []) or []
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    prov_id = args.get("provisional_id")

                    # resolve any potential provisional IDs in class_ids
                    cids_real = [provisional_to_real.get(cid, cid) for cid in cids_raw]
                    valid_cids = [cid for cid in cids_real if cid in all_classes]

                    if len(valid_cids) < 2:
                        decisions.append({
                            "action": "merge_skip_too_few",
                            "requested_class_ids": cids_raw,
                            "valid_class_ids": valid_cids,
                            "justification": justification,
                            "confidence": confidence_val
                        })
                        continue

                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)

                    # map provisional id -> new class id
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid
                    # also map old real ids -> new id so later references can still resolve
                    for old in valid_cids:
                        provisional_to_real.setdefault(old, new_cid)

                    decisions.append({
                        "action": "merge_classes",
                        "input_class_ids": valid_cids,
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)

                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "create_class",
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "member_ids_added": mids_valid,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    eids_valid = [e for e in eids if e in id_to_entity]

                    from_c = resolve_class_id(from_c_raw, all_classes, provisional_to_real, allow_missing=True)
                    to_c = resolve_class_id(to_c_raw, all_classes, provisional_to_real, allow_missing=False)

                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)

                    decisions.append({
                        "action": "reassign_entities",
                        "entity_ids": eids_valid,
                        "from": from_c_raw,
                        "from_resolved": from_c,
                        "to": to_c_raw,
                        "to_resolved": to_c,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")
                    new_remark = args.get("remark")

                    cid_real = resolve_class_id(cid_raw, all_classes, provisional_to_real, allow_missing=False)
                    execute_modify_class(all_classes, cid_real, new_name, new_desc, new_type, new_group, new_remark)

                    decisions.append({
                        "action": "modify_class",
                        "class_id": cid_raw,
                        "class_id_resolved": cid_real,
                        "new_name": new_name,
                        "new_description": new_desc,
                        "new_class_type_hint": new_type,
                        "new_class_group": new_group,
                        "remark": new_remark,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "split_class":
                    source_raw = args.get("source_class_id")
                    splits_specs = args.get("splits", []) or []

                    source_real = resolve_class_id(source_raw, all_classes, provisional_to_real, allow_missing=False)
                    created = execute_split_class(all_classes, source_real, splits_specs)

                    created_summary = []
                    for new_cid, prov_id, mids_used in created:
                        if prov_id:
                            provisional_to_real[prov_id] = new_cid
                        created_summary.append({
                            "new_class_id": new_cid,
                            "provisional_id": prov_id,
                            "member_ids": mids_used
                        })

                    decisions.append({
                        "action": "split_class",
                        "source_class_id": source_raw,
                        "source_class_id_resolved": source_real,
                        "created_classes": created_summary,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                else:
                    decisions.append({
                        "action": "skip_unknown_function",
                        "raw": step,
                        "justification": justification,
                        "confidence": confidence_val
                    })

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step,
                    "justification": justification,
                    "confidence": confidence_val
                })

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

if __name__ == "__main__":
    classres_main()

#endregion#? Cls Res V6
#?#########################  End  ##########################








#?######################### Start ##########################
#region:#?   Cls Res V7  - Split + Remark + Summary

#!/usr/bin/env python3
"""
classres_iterative_v7.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify/split) for each cluster,
then execute those functions locally and produce final resolved classes.

Key features:
- TWO-LAYER SCHEMA: Class_Group -> Classes -> Entities
- class_label treated as provisional; may be revised if evidence suggests a clearer name.
- LLM orders structural + schema actions using a small function vocabulary.
- Provisional IDs for newly created/merged/split classes so later steps can refer to them.
- 'remarks' channel so the LLM can flag out-of-scope or higher-level concerns without
  misusing structural functions.
- Conservative behavior with strong validation and logging.
- Summary folder with aggregated decisions and useful statistics.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
  - summary/all_clusters_decisions.json (aggregated decisions)
  - summary/stats_summary.json (aggregate statistics)
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
#INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res-Wrong.json")
SRC_ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI
OPENAI_MODEL = "gpt-4.1"  # adjust as needed
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 3000
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# behavioral flags
VERBOSE = True
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE:
            print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label", ""))[:120])
        descs.append(safe_str(c.get("class_description", ""))[:300])
        types.append(safe_str(c.get("class_type_hint", ""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt", ""))[:200])
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name", ""))
            desc = safe_str(m.get("entity_description", ""))
            etype = safe_str(m.get("entity_type_hint", ""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str, float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    emb_evid = embedder.encode_batch(evids) if any(t.strip() for t in evids) else None
    emb_mem = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label)
    emb_desc = ensure(emb_desc)
    emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid)
    emb_mem = ensure(emb_mem)

    w_label = weights.get("label", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_type = weights.get("type_hint", 0.0)
    w_evid = weights.get("evidence", 0.0)
    w_mem = weights.get("members", 0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0:
        raise ValueError("invalid class emb weights")
    w_label /= W
    w_desc /= W
    w_type /= W
    w_evid /= W
    w_mem /= W

    combined = (
        w_label * emb_label
        + w_desc * emb_desc
        + w_type * emb_type
        + w_evid * emb_evid
        + w_mem * emb_mem
    )
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP
) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP
    if use_umap and UMAP_AVAILABLE and N >= 6:
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, n_comp={safe_n_components}, n_nei={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template (extended) ------------------
CLSRES_PROMPT_TEMPLATE = """
You are a very proactive class-resolution assistant.
You are given a set of candidate CLASSES that appear to belong, or may plausibly belong, to the same semantic cluster.
Your job is to produce a clear, *actionable* ordered list of schema edits for the given cluster.
Do NOT be passive. If the evidence supports structural change (merge / split / reassign / create), you MUST propose it.

Only return [] if there is extremely strong evidence that NO change is needed (rare).

The cluster grouping you are given is only *suggestive* and may be incorrect — your job is to resolve, correct, and produce a coherent schema from the evidence.

This is an iterative process — act now with well-justified structural corrections (include a short justification), rather than deferring small but meaningful fixes.
This step will be repeated in later iterations; reasonable but imperfect changes can be corrected later.
It is worse to miss a necessary change than to propose a well-justified change that might be slightly adjusted later.

Your CRUCIAL task is to refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- class_label:
  Existing class_label values are PROVISIONAL names.
  You MAY revise them when entity evidence suggests a clearer or a better canonical label.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

- remarks (optional, internal):
  Free-text notes attached to a class, used to flag important issues that are
  outside the scope of structural changes (e.g., suspected entity-level duplicates
  that should be checked elsewhere).

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

Note: the provided cluster grouping is tentative and may be wrong — 
you must correct it as needed to produce a coherent Class_Group → Class → Entity schema.

For the given cluster of classes:

1) Assess whether any structural changes are REQUIRED:
   - merge duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class when necessary
   - split an overloaded class into more coherent subclasses when justified
   - modify class metadata when meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct (which is not the case most of the time), assigning or confirming Class_Group ALONE is sufficient.

3) If you notice important issues that are OUTSIDE the scope of these functions
   (e.g., upstream entity resolution that you suspect is wrong, or entities that
   look identical but should not be changed here), DO NOT try to fix them via merge or reassign.
   Instead, attach a human-facing remark via modify_class (using the 'remark' field)
   so that a human can review it later.

========================
SOME CONSERVATISM RULES (They should not make you passive)
========================

- Always try to attempt structural proposals (merge/split/reassign/create) unless the cluster truly is already optimal.
- Do NOT perform cosmetic edits or unnecessary normalization.

You MAY perform multiple structural actions in one cluster
(e.g., merge + rename + reassign + split), when needed.

========================
MERGING & OVERLAP HEURISTICS
========================

- Entity overlap alone does not automatically require merging.
- Merge when evidence indicates the SAME underlying concepts
  (e.g., near-identical semantics, interchangeable usage, or redundant distinctions).
- Reassignment is appropriate when overlap reveals mis-typed or mis-scoped entities,
  even if classes should remain separate.

Quick heuristic:
- Same concept → merge.
- Different concept, same domain → keep separate classes with the SAME Class_Group.
- Different domain → different Class_Group.

Avoid vague Class_Group names (e.g., "Misc", "General", "Other").
Prefer domain-meaningful groupings that help connect related classes.

If a class should be collapsed or weakened but has no clear merge partner, DO NOT use merge_classes;
instead, reassign its entities to better classes and/or add a remark for human review.

IMPORTANT:
- You MUST NOT call merge_classes with only one class_id.
- merge_classes is ONLY for merging TWO OR MORE existing classes into ONE new class.
- If you only want to update or clarify a single class (for example, its label, description,
  type hint, class_group, or remarks), use modify_class instead.
- Do NOT use merge_classes to clean up or deduplicate entities inside one class.

========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class", "split_class"]
- "args": arguments as defined below.

ID HANDLING RULES
- You MUST NOT invent real class IDs.
- You MUST use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except when referring to newly merged/created/split classes.
- When you need to refer to a newly merged/created/split class in later steps,
  you MUST assign a provisional_id (any consistent string).
- Use the same provisional_id whenever referencing that new class again.
- After you merge classes into a new class, you should NOT continue to treat the original
  class_ids as separate entities; refer to the new merged class via its provisional_id.

Example (pattern, not required verbatim):

{
  "function": "merge_classes",
  "args": {
    "class_ids": ["ClsC_da991b68", "ClsC_e32f4a47"],
    "provisional_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "new_name": "...",
    "new_description": "...",
    "new_class_type_hint": "Standard",
    "justification": "One-line reason citing entity overlap and semantic equivalence.",
    "remark": null,
    "confidence": 0.95
  }
}

Later:

{
  "function": "reassign_entities",
  "args": {
    "entity_ids": ["En_xxx"],
    "from_class_id": "ClsC_e32f4a47",
    "to_class_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "justification": "Why this entity fits better in the merged class.",
    "remark": null,
    "confidence": 0.9
  }
}

We will internally map provisional_id → real class id.

------------------------
Function definitions
------------------------

JUSTIFICATION REQUIREMENT
- Every function call MUST include:
    "justification": "<one-line reason>"
  explaining why the action is necessary.
- This justification should cite concrete evidence (entity overlap, conflicting descriptions,
  mis-scoped members, missing Class_Group, overloaded classes, etc.).
- You MAY also include "confidence": <0.0–1.0> to indicate your belief in the action.
- You MAY include "remark" to provide a short human-facing note.

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,    # how you will refer to the new class later
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>],          # optional, must be from provided entities
  "provisional_id": <string or null>,    # how you will refer to this new class later
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

- Use modify_class with 'remark' (and no structural change) when you want to flag
  issues that are outside the scope of this step (e.g., suspected entity-level duplicates).

5) split_class
args = {
  "source_class_id": <existing_class_id or provisional_id>,
  "splits": [
    {
      "name": <string or null>,
      "description": <string or null>,
      "class_type_hint": <string or null>,
      "member_ids": [<entity_ids>],      # must be from source_class member_ids
      "provisional_id": <string or null> # how you will refer to this new split class
    },
    ...
  ],
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

Semantics of split_class:
- Use split_class when a single class is overloaded and should be divided into
  narrower, more coherent classes.
- Entities listed in splits[*].member_ids MUST come from the source_class's member_ids.
- The specified entities are REMOVED from the source_class and grouped into new classes.
- Any members not mentioned in any split remain in the source_class.

NOTE:
- Class_Group is normally set or updated via modify_class.
- Assigning or confirming Class_Group is also REQUIRED for every cluster unless it is already clearly correct.

========================
VALIDATION RULES
========================

- Use ONLY provided entity_ids and class_ids (candidate_id values) for existing classes.
- For new classes, use provisional_id handles and be consistent.
- Order matters: later steps may depend on earlier ones.
- merge_classes with fewer than 2 valid class_ids will be ignored.

========================
STRATEGY GUIDANCE
========================

- Prefer assigning/adjusting Class_Group over heavy structural changes when both are valid.
- Merge classes ONLY when they are genuinely redundant (same concept).
- Use split_class when a class clearly bundles multiple distinct concepts that should be separated.
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity and meaningful structure, not cosmetic cleanup.
- If in doubt about structural change but you see a potential issue, use modify_class with a 'remark'
  rather than forcing an uncertain structural edit.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)
- remarks (optional list of prior remarks)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] only if you are highly confident (> very strong evidence) that no change is needed.
If any ambiguous or conflicting evidence exists, return a concrete ordered action list
(with justifications and, optionally, confidence scores).
You are a very proactive class-resolution assistant. You are not called very proactive only by using modify_class for new_class_group. You must use other functions as well to be called a proactive class-resolution assistant.

"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # find first [ ... ] block
    start = s.find("[")
    end = s.rfind("]")
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end + 1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(
    all_classes: Dict[str, Dict],
    class_ids: List[str],
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str]
) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if len(class_ids) < 2:
        raise ValueError("merge_classes: need at least 2 valid class_ids")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members_map: Dict[str, Dict] = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None

    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence", 0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep first; fixable later by modify_class
                class_group_choice = class_group_choice
        for m in c.get("members", []):
            members_map[m["id"]] = m

    if new_desc:
        desc_choice = new_desc
    new_label = new_name or all_classes[class_ids[0]].get("class_label", "MergedClass")
    if not new_type:
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break

    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "remarks": [],
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(
    all_classes: Dict[str, Dict],
    name: str,
    description: Optional[str],
    class_type_hint: Optional[str],
    member_ids: Optional[List[str]],
    id_to_entity: Dict[str, Dict]
) -> str:
    if not name:
        raise ValueError("create_class: 'name' is required")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "remarks": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict]
):
    # remove from source(s)
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence", 0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(
    all_classes: Dict[str, Dict],
    class_id: str,
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str],
    new_class_group: Optional[str],
    new_remark: Optional[str]
):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    if new_remark:
        existing = c.get("remarks")
        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [str(existing)]
        existing.append(str(new_remark))
        c["remarks"] = existing
    all_classes[class_id] = c

def execute_split_class(
    all_classes: Dict[str, Dict],
    source_class_id: str,
    splits_specs: List[Dict[str, Any]]
) -> List[Tuple[str, Optional[str], List[str]]]:
    """
    Split a source class into several new classes.
    Returns list of (new_class_id, provisional_id, member_ids_used) for each created class.
    """
    if source_class_id not in all_classes:
        raise ValueError(f"split_class: source_class_id {source_class_id} not found")
    src = all_classes[source_class_id]
    src_members = src.get("members", []) or []
    src_member_map = {m["id"]: m for m in src_members if isinstance(m, dict) and m.get("id")}
    used_ids: set = set()
    created: List[Tuple[str, Optional[str], List[str]]] = []

    for spec in splits_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        desc = spec.get("description")
        th = spec.get("class_type_hint") or src.get("class_type_hint", "")
        mids_raw = spec.get("member_ids", []) or []
        prov_id = spec.get("provisional_id")
        # use only members that belong to source and are not already used
        valid_mids = []
        for mid in mids_raw:
            if mid in src_member_map and mid not in used_ids:
                valid_mids.append(mid)
        if not valid_mids:
            continue
        members = [src_member_map[mid] for mid in valid_mids]
        new_cid = "ClsR_" + uuid.uuid4().hex[:8]
        obj = {
            "candidate_id": new_cid,
            "class_label": name or src.get("class_label", "SplitClass"),
            "class_description": desc or src.get("class_description", ""),
            "class_type_hint": th or src.get("class_type_hint", ""),
            "class_group": src.get("class_group", "TBD"),
            "confidence": src.get("confidence", 0.5),
            "evidence_excerpt": src.get("evidence_excerpt", ""),
            "member_ids": valid_mids,
            "members": members,
            "candidate_ids": [],
            "remarks": [],
            "_split_from": source_class_id,
            "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        all_classes[new_cid] = obj
        used_ids.update(valid_mids)
        created.append((new_cid, prov_id, valid_mids))

    if used_ids:
        remaining_members = [m for m in src_members if m["id"] not in used_ids]
        src["members"] = remaining_members
        src["member_ids"] = [m["id"] for m in remaining_members]
        all_classes[source_class_id] = src

    return created

# helper to resolve real class ID (handles provisional IDs and retired IDs)
def resolve_class_id(
    raw_id: Optional[str],
    all_classes: Dict[str, Dict],
    provisional_to_real: Dict[str, str],
    allow_missing: bool = False
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes")
    return real

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members)
    id_to_entity: Dict[str, Dict] = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys, class_group field, and remarks field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        members = c.get("members", []) or []
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        # normalize remarks
        if "remarks" not in c or c["remarks"] is None:
            c["remarks"] = []
        elif not isinstance(c["remarks"], list):
            c["remarks"] = [str(c["remarks"])]
        c["candidate_id"] = cid
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end

    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")

        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label", ""),
                "class_description": c.get("class_description", ""),
                "class_type_hint": c.get("class_type_hint", ""),
                "class_group": c.get("class_group", "TBD"),
                "confidence": float(c.get("confidence", 0.0)),
                "evidence_excerpt": c.get("evidence_excerpt", ""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact,
                "remarks": c.get("remarks", [])
            })

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            continue

        # mapping from provisional_id (and retired ids) -> real class id (per cluster)
        provisional_to_real: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}

            justification = args.get("justification")
            confidence_val = args.get("confidence", None)
            remark_val = args.get("remark")

            try:
                if fn == "merge_classes":
                    cids_raw = args.get("class_ids", []) or []
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    prov_id = args.get("provisional_id")

                    # resolve any potential provisional IDs in class_ids
                    cids_real = [provisional_to_real.get(cid, cid) for cid in cids_raw]
                    valid_cids = [cid for cid in cids_real if cid in all_classes]

                    if len(valid_cids) < 2:
                        decisions.append({
                            "action": "merge_skip_too_few",
                            "requested_class_ids": cids_raw,
                            "valid_class_ids": valid_cids,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)

                    # map provisional id -> new class id
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid
                    # also map old real ids -> new id so later references can still resolve
                    for old in valid_cids:
                        provisional_to_real.setdefault(old, new_cid)

                    decisions.append({
                        "action": "merge_classes",
                        "input_class_ids": valid_cids,
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)

                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "create_class",
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "member_ids_added": mids_valid,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    eids_valid = [e for e in eids if e in id_to_entity]

                    from_c = resolve_class_id(from_c_raw, all_classes, provisional_to_real, allow_missing=True)
                    to_c = resolve_class_id(to_c_raw, all_classes, provisional_to_real, allow_missing=False)

                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)

                    decisions.append({
                        "action": "reassign_entities",
                        "entity_ids": eids_valid,
                        "from": from_c_raw,
                        "from_resolved": from_c,
                        "to": to_c_raw,
                        "to_resolved": to_c,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")
                    new_remark = args.get("remark")

                    cid_real = resolve_class_id(cid_raw, all_classes, provisional_to_real, allow_missing=False)
                    execute_modify_class(all_classes, cid_real, new_name, new_desc, new_type, new_group, new_remark)

                    decisions.append({
                        "action": "modify_class",
                        "class_id": cid_raw,
                        "class_id_resolved": cid_real,
                        "new_name": new_name,
                        "new_description": new_desc,
                        "new_class_type_hint": new_type,
                        "new_class_group": new_group,
                        "remark": new_remark,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "split_class":
                    source_raw = args.get("source_class_id")
                    splits_specs = args.get("splits", []) or []

                    source_real = resolve_class_id(source_raw, all_classes, provisional_to_real, allow_missing=False)
                    created = execute_split_class(all_classes, source_real, splits_specs)

                    created_summary = []
                    for new_cid, prov_id, mids_used in created:
                        if prov_id:
                            provisional_to_real[prov_id] = new_cid
                        created_summary.append({
                            "new_class_id": new_cid,
                            "provisional_id": prov_id,
                            "member_ids": mids_used
                        })

                    decisions.append({
                        "action": "split_class",
                        "source_class_id": source_raw,
                        "source_class_id_resolved": source_real,
                        "created_classes": created_summary,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                else:
                    decisions.append({
                        "action": "skip_unknown_function",
                        "raw": step,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step,
                    "justification": justification,
                    "remark": remark_val,
                    "confidence": confidence_val
                })

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

    # ---------------------- SUMMARY AGGREGATION -------------------------
    summary_dir = OUT_DIR / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Aggregate all per-cluster decision files
    cluster_decisions: List[Dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("cluster_*_decisions.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            cluster_decisions.append(obj)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")

    # Write aggregated decisions (same per-cluster structure, just as a list)
    all_clusters_decisions_path = summary_dir / "all_clusters_decisions.json"
    all_clusters_decisions_path.write_text(
        json.dumps(cluster_decisions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Compute statistics
    total_clusters = len(cluster_decisions)
    actions_by_type: Dict[str, int] = {}
    total_remarks = 0
    clusters_with_any_decisions = 0
    clusters_with_structural = 0
    clusters_only_classgroup = 0

    structural_actions = {"merge_classes", "create_class", "reassign_entities", "split_class"}

    for cd in cluster_decisions:
        decs = cd.get("executed_decisions", [])
        if not decs:
            continue
        clusters_with_any_decisions += 1
        has_structural = False
        only_classgroup = True

        for d in decs:
            act = d.get("action")
            actions_by_type[act] = actions_by_type.get(act, 0) + 1

            # count remarks if present
            rem = d.get("remark")
            if rem:
                total_remarks += 1

            if act in structural_actions:
                has_structural = True
                only_classgroup = False
            elif act == "modify_class":
                new_group = d.get("new_class_group")
                new_name = d.get("new_name")
                new_desc = d.get("new_description")
                new_type = d.get("new_class_type_hint")
                # "only class group" means the only change is class_group (no name/desc/type/remark change)
                # remark alone is allowed
                if not new_group or new_name or new_desc or new_type:
                    only_classgroup = False
            else:
                # merge_skip_too_few, error_executing, skip_unknown_function, etc.
                only_classgroup = False

        if has_structural:
            clusters_with_structural += 1
        if only_classgroup:
            clusters_only_classgroup += 1

    total_structural_actions = sum(
        count for act, count in actions_by_type.items() if act in structural_actions
    )
    total_errors = actions_by_type.get("error_executing", 0)

    stats = {
        "total_clusters": total_clusters,
        "total_clusters_with_any_decisions": clusters_with_any_decisions,
        "total_clusters_with_structural_changes": clusters_with_structural,
        "total_clusters_only_class_group_updates": clusters_only_classgroup,
        "total_actions_by_type": actions_by_type,
        "total_structural_actions": total_structural_actions,
        "total_errors": total_errors,
        "total_remarks_logged": total_remarks,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = summary_dir / "stats_summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] summary decisions -> {all_clusters_decisions_path}")
    print(f"[done] summary stats -> {stats_path}")

if __name__ == "__main__":
    classres_main()

#endregion#? Cls Res V7
#?#########################  End  ##########################











#?######################### Start ##########################
#region:#?   Multi Run Cls Res

# =========================
# Iterative runner for ClassRes (UPDATED final-summary exports)
# Paste this into the same file after V7 (replaces prior runner's final summary block)
# =========================

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------
# CONFIG - reuse or override
# -----------------------
BASE_INPUT_CLASSES = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json")
EXPERIMENT_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res_IterativeRuns")

MAX_RUNS: int = 3
STRUCTURAL_CHANGE_THRESHOLD: Optional[int] = 0
TOTAL_ACTIONS_THRESHOLD: Optional[int] = None
MAX_NO_CHANGE_RUNS: Optional[int] = 1

FINAL_CLASSES_FILENAME = "final_classes_resolved.json"
ACTION_LOG_FILENAME = "cls_res_action_log.jsonl"

# -----------------------
# Helpers (same as before)
# -----------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _safe_json_load_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def compute_run_summary_from_action_log(action_log_path: Path) -> Dict[str, Any]:
    summary = {
        "total_clusters": 0,
        "total_clusters_with_any_decisions": 0,
        "total_clusters_with_structural_changes": 0,
        "total_clusters_only_class_group_updates": 0,
        "total_actions_by_type": {},
        "total_structural_actions": 0,
        "total_errors": 0,
        "total_remarks_logged": 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    if not action_log_path.exists():
        return summary

    with action_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _safe_json_load_line(line)
            if not obj:
                continue
            summary["total_clusters"] += 1
            executed = obj.get("executed_decisions", []) or []
            if executed:
                summary["total_clusters_with_any_decisions"] += 1

            structural_here = 0
            only_class_group_updates = True
            remarks_here = 0

            for entry in executed:
                action = entry.get("action")
                summary["total_actions_by_type"].setdefault(action, 0)
                summary["total_actions_by_type"][action] += 1

                if action in ("merge_classes", "create_class", "reassign_entities", "split_class"):
                    structural_here += 1
                    only_class_group_updates = False
                elif action == "modify_class":
                    new_name = entry.get("new_name")
                    new_desc = entry.get("new_description")
                    new_type = entry.get("new_class_type_hint")
                    new_group = entry.get("new_class_group")
                    remark = entry.get("remark") or None
                    if remark:
                        remarks_here += 1
                    if any([new_name, new_desc, new_type]):
                        only_class_group_updates = False
                elif action == "error_executing":
                    summary["total_errors"] += 1
                    only_class_group_updates = False
                # count explicit remark fields
                if isinstance(entry, dict) and entry.get("remark"):
                    remarks_here += 1

            summary["total_structural_actions"] += structural_here
            if structural_here > 0:
                summary["total_clusters_with_structural_changes"] += 1
            if only_class_group_updates and executed:
                summary["total_clusters_only_class_group_updates"] += 1
            summary["total_remarks_logged"] += remarks_here

    return summary

# -----------------------
# New: overall export helpers
# -----------------------
def _build_entity_lookup_from_input(input_path: Path) -> Dict[str, Dict]:
    """
    Build map entity_id -> full entity object from the original input classes file.
    This lets us recover fields like chunk_id, resolution_context when final classes may
    only contain compact member info.
    """
    entity_map: Dict[str, Dict] = {}
    if not input_path.exists():
        return entity_map
    try:
        arr = _load_json(input_path)
        for c in arr:
            for m in c.get("members", []) or []:
                if isinstance(m, dict) and m.get("id"):
                    entity_map[m["id"]] = m
    except Exception:
        # best-effort: ignore on error
        pass
    return entity_map

def _aggregate_final_classes_and_entities(final_classes_path: Path, base_input_path: Path) -> Dict[str, Any]:
    """
    Return dict with:
      - classes: final classes array (as in final_classes_resolved.json)
      - entities_map: map entity_id -> merged full entity info (pull fields from classes and original input)
    """
    classes = []
    entities_map: Dict[str, Dict] = {}
    if final_classes_path.exists():
        try:
            classes = _load_json(final_classes_path)
        except Exception:
            classes = []

    # build lookup from original input to backfill entity fields
    input_entity_lookup = _build_entity_lookup_from_input(base_input_path)

    # iterate classes and collect members, merge any available entity fields
    for c in classes:
        members = c.get("members", []) or []
        for m in members:
            mid = m.get("id") if isinstance(m, dict) else None
            if not mid:
                continue
            # start with the member object from final classes (may be compact)
            merged = dict(m) if isinstance(m, dict) else {"id": mid}
            # if original input had richer info, copy missing fields
            original = input_entity_lookup.get(mid)
            if original:
                for k, v in original.items():
                    if k not in merged or merged.get(k) in (None, "", []):
                        merged[k] = v
            # attach class pointers (class id, label, class_group)
            merged["_class_id"] = c.get("candidate_id") or c.get("candidate_id") or None
            merged["_class_label"] = c.get("class_label")
            merged["_class_group"] = c.get("class_group")
            entities_map[mid] = merged

    # also include any entities that were in the input but not present in final classes
    for mid, original in input_entity_lookup.items():
        if mid not in entities_map:
            merged = dict(original)
            merged["_class_id"] = None
            merged["_class_label"] = None
            merged["_class_group"] = None
            entities_map[mid] = merged

    return {"classes": classes, "entities_map": entities_map}

def _write_entities_with_class_jsonl(entities_map: Dict[str, Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for eid, ent in entities_map.items():
            rec = {
                "entity_id": eid,
                "entity": ent,
                "class_id": ent.get("_class_id"),
                "class_label": ent.get("_class_label"),
                "class_group": ent.get("_class_group")
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -----------------------
# Main runner (same loop as before, but calls enhanced final exporter)
# -----------------------
def run_pipeline_iteratively():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    overall_runs: List[Dict[str, Any]] = []
    no_change_streak = 0
    current_input_path = BASE_INPUT_CLASSES

    for run_idx in range(MAX_RUNS):
        print("\n" + "=" * 36)
        print(f"=== RUN {run_idx:02d} ===")
        print("=" * 36)

        run_dir = EXPERIMENT_ROOT / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # set globals used by your pipeline
        globals()["INPUT_CLASSES"] = current_input_path
        globals()["OUT_DIR"] = run_dir
        globals()["RAW_LLM_DIR"] = run_dir / "llm_raw"
        globals()["RAW_LLM_DIR"].mkdir(parents=True, exist_ok=True)

        # call the pipeline's main function (assumes classres_main defined already)
        print(f"[run {run_idx}] calling classres_main() with INPUT_CLASSES={current_input_path} OUT_DIR={run_dir}")
        classres_main()

        final_classes_path = run_dir / FINAL_CLASSES_FILENAME
        action_log_path = run_dir / ACTION_LOG_FILENAME

        run_summary = compute_run_summary_from_action_log(action_log_path)
        run_summary["run_index"] = run_idx
        run_summary["run_path"] = str(run_dir)
        run_summary["final_classes_path"] = str(final_classes_path) if final_classes_path.exists() else None

        summary_dir = run_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        run_summary_path = summary_dir / "cls_res_summary.json"
        _write_json(run_summary_path, run_summary)

        overall_runs.append({"run_index": run_idx, "run_dir": str(run_dir), "summary_path": str(run_summary_path)})

        total_structural = int(run_summary.get("total_structural_actions", 0))
        total_actions = int(sum(run_summary.get("total_actions_by_type", {}).values() or []))

        print(f"[run {run_idx}] total_structural_actions = {total_structural}")
        print(f"[run {run_idx}] total_actions = {total_actions}")

        is_no_change = False
        if STRUCTURAL_CHANGE_THRESHOLD is not None and total_structural <= STRUCTURAL_CHANGE_THRESHOLD:
            is_no_change = True
        if TOTAL_ACTIONS_THRESHOLD is not None and total_actions <= TOTAL_ACTIONS_THRESHOLD:
            is_no_change = True

        if MAX_NO_CHANGE_RUNS is not None:
            if is_no_change:
                no_change_streak += 1
            else:
                no_change_streak = 0

            print(f"[run {run_idx}] no_change_streak = {no_change_streak} (threshold {MAX_NO_CHANGE_RUNS})")
            if no_change_streak >= MAX_NO_CHANGE_RUNS:
                print(f"[stop] Convergence achieved after run {run_idx} (no_change_streak={no_change_streak}).")
                current_input_path = final_classes_path if final_classes_path.exists() else current_input_path
                break

        if final_classes_path.exists():
            current_input_path = final_classes_path
        else:
            print(f"[warn] final classes file not found for run {run_idx}. Stopping iterative runs.")
            break

    # -----------------------
    # ENHANCED FINAL EXPORTS
    # -----------------------
    overall_dir = EXPERIMENT_ROOT / "overall_summary"
    overall_dir.mkdir(parents=True, exist_ok=True)

    # collect per-run summaries (read the run summary JSON files)
    per_run_stats: List[Dict[str, Any]] = []
    for r in overall_runs:
        sp = Path(r["summary_path"])
        if sp.exists():
            try:
                per_run_stats.append(_load_json(sp))
            except Exception:
                per_run_stats.append({"run_index": r["run_index"], "error": "failed to load summary"})

    # aggregate overall stats (summing counts)
    aggregated = {
        "total_runs_executed": len(per_run_stats),
        "sum_total_clusters": sum([p.get("total_clusters", 0) for p in per_run_stats]),
        "sum_structural_actions": sum([p.get("total_structural_actions", 0) for p in per_run_stats]),
        "sum_errors": sum([p.get("total_errors", 0) for p in per_run_stats]),
        "sum_remarks": sum([p.get("total_remarks_logged", 0) for p in per_run_stats]),
        "by_run": per_run_stats,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = overall_dir / "stats.json"
    _write_json(stats_path, aggregated)

    # Build classes + entities bundle for downstream relation stage
    final_classes = []
    if current_input_path.exists():
        try:
            final_classes = _load_json(current_input_path)
        except Exception:
            final_classes = []

    classes_and_entities = _aggregate_final_classes_and_entities(current_input_path, BASE_INPUT_CLASSES)
    classes_and_entities_path = overall_dir / "classes_and_entities.json"
    _write_json(classes_and_entities_path, classes_and_entities)

    # write entities+class mapping as jsonl for downstream tools
    entities_with_class_path = overall_dir / "entities_with_class.jsonl"
    _write_entities_with_class_jsonl(classes_and_entities.get("entities_map", {}), entities_with_class_path)

    # also write the final_classes array alone for convenience
    final_classes_path_out = overall_dir / "final_classes_resolved.json"
    _write_json(final_classes_path_out, final_classes)

    print(f"\n[done] Overall stats written to: {stats_path}")
    print(f"[done] Classes+entities bundle written to: {classes_and_entities_path}")
    print(f"[done] Entities jsonl (entity + class info) written to: {entities_with_class_path}")
    print(f"[done] Final classes (copy) written to: {final_classes_path_out}")

# -----------------------
# To run: call run_pipeline_iteratively() after pasting this block.
# -----------------------



#endregion#? Multi Run Cls Res
#?#########################  End  ##########################










#endregion#! Class Identification
#!#############################################  End Chapter  ##################################################











#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################






















#?######################### Start ##########################
#region:#?   Class Recognition

#endregion#? Class Recognition
#?#########################  End  ##########################











#?######################### Start ##########################
#region:#?   Relation Recognition


#endregion#? Relation Recognition
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   KG Assembly



#endregion#? KG Assembly
#?#########################  End  ##########################






#endregion:#!   SGCE-KG Generator
#!#############################################  End Chapter  ##################################################






























#!############################################# Start Chapter ##################################################
#region:#!   Evaluation

#endregion#! Evaluation
#!#############################################  End Chapter  ##################################################






#!############################################# Start Chapter ##################################################
#region:#!   Experiments

#endregion#! Experiments
#!#############################################  End Chapter  ##################################################










#!############################################# Start Chapter ##################################################
#region:#!   

#endregion#! 
#!#############################################  End Chapter  ##################################################

















#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################






















#?######################### Start ##########################
#region:#?   OpenAi API


#endregion#? OpenAi API
#?#########################  End  ##########################








