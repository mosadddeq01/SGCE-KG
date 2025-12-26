







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




#endregion#! Old Ent Res Manual Re Run
#!#############################################  End Chapter  ##################################################

























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










#endregion#! Entity Identification
#!#############################################  End Chapter  ##################################################







#!############################################# Start Chapter ##################################################
#region:#!   Class Identification






#?######################### Start ##########################
#region:#?    Embed_and_cluster_final_entities


#!/usr/bin/env python3
"""
embed_and_cluster_final_entities.py

- INPUT: final_entities_master.jsonl (your consolidated, resolved entities)
- ACTIONS:
    1) Build weighted text per-entity using fields: name, desc, ctx, type and WEIGHTS
    2) Embed each field using an HF model (default: BAAI/bge-large-en-v1.5) and combine with weights
    3) Optionally run UMAP reduction
    4) Cluster combined vectors with HDBSCAN (variable K / density-based — no fixed K)
    5) Save:
        - clustered entities JSONL with `_cluster_id` field under:
          /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/ResEnt_Clustering/entities_clustered_final.jsonl
        - clusters summary JSON under same folder
        - combined embeddings .npy (same folder) for later nearest-neighbor retrieval
        - optionally build a FAISS index file (if faiss installed)
- NOTE: This script is self-contained and follows the patterns in your repo.
- Adjust the CONFIG below as needed.

Author: adapted to your pipeline and requests
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import uuid
import sys

# Transformer / torch embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# Clustering and reduction
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`") from e

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# Optional: faiss (for fast nearest neighbor retrieval later)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ------------------ CONFIG ------------------
# input final entities (use your FinalCompAnalysis master)
ENTITIES_IN = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Overall_After_Two_Run/FinalCompAnalysis/final_entities_master.jsonl"

# output folder (as requested)
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/ResEnt_Clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTERED_JSONL = OUT_DIR / "entities_clustered_final.jsonl"
OUT_CLUSTERS_SUMMARY = OUT_DIR / "clusters_summary_final.json"
OUT_EMBEDDINGS_NPY = OUT_DIR / "entities_embeds_combined.npy"
OUT_FAISS_INDEX = OUT_DIR / "entities_faiss.index"  # optional, only if faiss available

# Embedding model (change if you prefer)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# weights you specified (we will normalize them inside code)
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}

# Batching
BATCH_SIZE = 32

# HDBSCAN params (force density-based clusters; user asked variable-K)
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # embeddings will be normalized so euclidean ~ cosine

# UMAP params (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

# Save intermediate per-field embeddings? (saves memory but may be large)
SAVE_FIELD_EMBS = False

# ------------------ Helpers ------------------
def load_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, list) or isinstance(v, dict):
        # preserve some structure but short
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)

# ------------------ HF embedder (mean pooling) ------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = BATCH_SIZE, max_length: int = 1024) -> np.ndarray:
        """
        Encode list of texts to numpy (N, D). Uses mean pooling over last_hidden_state.
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state  # (B, L, D)
            pooled = mean_pool(token_embeds, attention_mask)  # (B, D)
            emb = pooled.cpu().numpy()
            all_embs.append(emb)
        embs = np.vstack(all_embs)
        # L2 normalize
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Field builder & combined embedding ------------------
def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str]]:
    names, descs, ctxs, types = [], [], [], []
    for e in entities:
        # prefer canonical fields from your master file
        name = safe_text(e, "entity_name") or safe_text(e, "canonical_name") or ""
        desc = safe_text(e, "entity_description") or ""
        # context: try chunk_text_truncated, text_span, used_context_excerpt, example_member_spans
        ctx = safe_text(e, "chunk_text_truncated") or safe_text(e, "text_span") or safe_text(e, "used_context_excerpt") or safe_text(e, "example_member_spans") or ""
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "canonical_type") or ""
        names.append(name)
        descs.append(desc)
        ctxs.append(ctx)
        types.append(etype)
    return names, descs, ctxs, types

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights: Dict[str, float]) -> np.ndarray:
    names, descs, ctxs, types = build_field_texts(entities)
    D_ref = None

    # encode each present field; if a field is entirely empty, treat as zeros
    emb_name = embedder.encode_texts(names) if any(t.strip() for t in names) else None
    if emb_name is not None:
        D_ref = emb_name.shape[1]
    emb_desc = embedder.encode_texts(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]
    emb_ctx = embedder.encode_texts(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]
    emb_type = embedder.encode_texts(types) if any(t.strip() for t in types) else None
    if D_ref is None and emb_type is not None:
        D_ref = emb_type.shape[1]

    if D_ref is None:
        raise ValueError("All entity fields empty — cannot embed")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref), dtype=np.float32)
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr.astype(np.float32)

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx = _ensure(emb_ctx)
    emb_type = _ensure(emb_type)

    # normalize provided weights to sum to 1
    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    w_type = weights.get("type", 0.0)
    Wsum = w_name + w_desc + w_ctx + w_type
    if Wsum <= 0:
        raise ValueError("Sum of WEIGHTS must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum; w_type /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx) + (w_type * emb_type)
    combined = normalize(combined, axis=1)
    return combined.astype(np.float32)

# ------------------ Clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = USE_UMAP,
                umap_dims: int = UMAP_N_COMPONENTS) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    X = embeddings
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] UMAP not available — running HDBSCAN on original embeddings")
        else:
            print(f"[cluster] running UMAP -> {umap_dims} dims")
            reducer = umap.UMAP(n_components=umap_dims, n_neighbors=UMAP_N_NEIGHBORS,
                                min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
            X = reducer.fit_transform(X)
            print("[cluster] UMAP done; reduced shape:", X.shape)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

# ------------------ Persistence helpers ------------------
def save_clustered_entities(entities: List[Dict], labels: np.ndarray, out_jsonl: Path, clusters_summary_path: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            rec = dict(e)
            rec["_cluster_id"] = int(lab)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # build simple summary
    summary = {}
    for idx, lab in enumerate(labels):
        summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
    clusters_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        json.dump({"n_entities": len(entities),
                   "n_clusters_including_noise": len(summary),
                   "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
    print(f"[save] wrote clustered entities to {out_jsonl} and summary to {clusters_summary_path}")

def save_embeddings(embeddings: np.ndarray, path: Path):
    np.save(str(path), embeddings)
    print(f"[save] saved embeddings to {path} (shape={embeddings.shape})")

def build_faiss_index(embeddings: np.ndarray, out_index_path: Path, use_gpu: bool = False):
    if not FAISS_AVAILABLE:
        print("[faiss] not available; skipping index build")
        return
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product since vectors normalized -> cosine
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print("[faiss] GPU init failed; falling back to CPU index:", e)
    index.add(embeddings)
    faiss.write_index(index if not use_gpu else faiss.index_gpu_to_cpu(index), str(out_index_path))
    print(f"[faiss] wrote index to {out_index_path}")

# ------------------ Main routine ------------------
def main(
    entities_in: str = ENTITIES_IN,
    out_dir: Path = OUT_DIR,
    weights: Dict[str, float] = WEIGHTS,
    embed_model: str = EMBED_MODEL,
    use_umap: bool = USE_UMAP,
    umap_dims: int = UMAP_N_COMPONENTS,
    hdb_min_cluster: int = HDBSCAN_MIN_CLUSTER_SIZE,
    hdb_min_samples: int = HDBSCAN_MIN_SAMPLES,
    save_faiss: bool = True
):
    print("[main] loading entities:", entities_in)
    entities = load_jsonl(entities_in)
    print(f"[main] loaded {len(entities)} entities")

    # instantiate embedder
    embedder = HFEmbedder(model_name=embed_model, device=DEVICE)

    # compute combined embeddings
    combined = compute_combined_embeddings(embedder, entities, weights=weights)
    print("[main] combined embeddings shape:", combined.shape)

    # save combined embeddings for later (nearest neighbor retrieval during class recognition)
    save_embeddings(combined, out_dir / OUT_EMBEDDINGS_NPY.name)

    # cluster with HDBSCAN (variable number of clusters)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=hdb_min_cluster,
                                          min_samples=hdb_min_samples,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=use_umap,
                                          umap_dims=umap_dims)
    labels_arr = np.array(labels)

    # diagnostics print
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}; noise: {n_noise} ({n_noise/n*100:.1f}%)")
    from collections import Counter
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    # save clustered entities + summary
    save_clustered_entities(entities, labels_arr, out_dir / OUT_CLUSTERED_JSONL.name, out_dir / OUT_CLUSTERS_SUMMARY.name)

    # optionally build FAISS index for later top-K retrieval (useful for class recognition)
    if save_faiss:
        build_faiss_index(combined, out_dir / OUT_FAISS_INDEX.name, use_gpu=False)

    print("[main] done.")


# ------------------ Safe CLI entry (replace your existing __main__ block) ------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Embed & HDBSCAN-cluster final entities (variable K).")
    parser.add_argument("--entities_in", type=str, default=ENTITIES_IN)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--use_umap", action="store_true", default=USE_UMAP)
    parser.add_argument("--umap_dims", type=int, default=UMAP_N_COMPONENTS)
    parser.add_argument("--hdb_min_cluster", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--hdb_min_samples", type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--no_faiss", action="store_true", help="Do not attempt to build a FAISS index")
    # Use parse_known_args so ipykernel / jupyter launcher args (like --f=...) are ignored
    args, unknown = parser.parse_known_args()

    # If we are inside an ipykernel (Jupyter), optionally override defaults for quick testing:
    in_ipykernel = False
    try:
        # ipykernel sets this module
        import ipykernel  # type: ignore
        in_ipykernel = True
    except Exception:
        in_ipykernel = False

    if in_ipykernel:
        print("[INFO] Running inside ipykernel / Jupyter. Unknown argv ignored:", unknown)
        # If you want notebook-friendly defaults different from CLI, set them here:
        # (they already default to ENTITIES_IN and OUT_DIR defined above)
        # Example (uncomment to override in notebook):
        # args.entities_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Overall_After_Two_Run/FinalCompAnalysis/final_entities_master.jsonl"
        # args.out_dir = str(OUT_DIR)
    else:
        # Running from terminal -> show any unknown args (should be none)
        if unknown:
            print("[WARN] Unknown args ignored:", unknown)

    # Ensure output dir exists
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Override module-level config variables with CLI values
    ENTITIES_IN = args.entities_in
    EMBED_MODEL = args.embed_model

    # Run main
    main(
        entities_in=ENTITIES_IN,
        out_dir=OUT_DIR,
        weights=WEIGHTS,
        embed_model=EMBED_MODEL,
        use_umap=bool(args.use_umap),
        umap_dims=int(args.umap_dims),
        hdb_min_cluster=int(args.hdb_min_cluster),
        hdb_min_samples=int(args.hdb_min_samples),
        save_faiss=not args.no_faiss
    )


#endregion#?  Embed_and_cluster_final_entities
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Utilities for Entity Identification


import json
from pathlib import Path

FINAL_ENTITIES = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Overall_After_Two_Run/FinalCompAnalysis/final_entities_master.jsonl"

def print_entities_by_name(name: str, case_sensitive: bool = False):
    p = Path(FINAL_ENTITIES)
    assert p.exists(), f"File not found: {p}"

    matches = []
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            row = json.loads(ln)
            ent_name = row.get("entity_name", "")

            if case_sensitive:
                ok = ent_name == name
            else:
                ok = ent_name.lower() == name.lower()

            if ok:
                matches.append(row)

    print(f"\nFound {len(matches)} matching rows for entity_name = '{name}'\n")
    for i, r in enumerate(matches, 1):
        print(f"--- Match {i} ---")
        print(json.dumps(r, ensure_ascii=False, indent=2))

# ---------------- Example ----------------
if __name__ == "__main__":
    print_entities_by_name(
        name="graphitization",
        case_sensitive=False
    )
    
    
    
    
    


path = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
x = "RT"

ids = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("entity_name") == x:
            ids.append(f'"{r.get("chunk_id")}"')

print(", ".join(ids))
 
    











import json

chunks_in = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl"
chunks_out = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence_TEST.jsonl"

keep_ids = {
    "Ch_000003","Ch_000038","Ch_000039","Ch_000040","Ch_000042","Ch_000043",
    "Ch_000045","Ch_000047","Ch_000048","Ch_000050","Ch_000051","Ch_000052",
    "Ch_000053","Ch_000054","Ch_000055","Ch_000060","Ch_000061","Ch_000062",
    "Ch_000064","Ch_000074","Ch_000075","Ch_000098","Ch_000099","Ch_000164",
    "Ch_000172","Ch_000173","Ch_000180","Ch_000192","Ch_000193","Ch_000210",
    "Ch_000222"
}

with open(chunks_in, "r", encoding="utf-8") as fin, \
     open(chunks_out, "w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        if r.get("id") in keep_ids:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Done.")







#endregion#? Utilities for Entity Identification
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   

#endregion#? 
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






















