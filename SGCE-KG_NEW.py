# import os
# print(os.getcwd())


# from pathlib import Path
# PROJECT_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG")


# # import os

# # PROJECT_ROOT = os.path.dirname(
# #     os.path.dirname(os.path.abspath(__file__))
# # )









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






#!############################################# Start Chapter ##################################################
#region:#!   SGCE-KG Generator




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



#?######################### Start ##########################
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
#?#########################  End  ##########################



#?######################### Start ##########################
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
#?#########################  End  ##########################



#?######################### Start ##########################
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
    150,# max_tokens_per_chunk -> None for fastest
    False,# keep_ref_text
    True, # strip_leading_headings
    True, # force
    False) # debug


#endregion#? Chunking v1
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
    "data/chunks_emb",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    False,   # use_small_model_for_dev
    32,     # batch_size
    None,   # device -> auto
    True,   # save_index
    True)  # force


#endregion#? Embedding + FAISS Index
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   Entity Recognition V6 - Broader hint better prmpting


import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG: paths ----------
CHUNKS_JSONL =  "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl"
ENTITIES_OUT = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl"
DEFAULT_DEBUG_DIR = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entity_raw_debug_prompts_outputs"

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
        "Your task is to extract entity mentions from the FOCUS chunk so they can be canonicalized later.",
        "",
        "PRINCIPLES (read carefully):",
        "- Be broad-minded: the list of type hints below is a helpful suggestion, but DO NOT be constrained by it.",
        "- If you believe a mention belongs to a different or more specific type, propose that type (string) in `entity_type_hint`.",
        "- If uncertain about a mention, still include it with a LOWER confidence score; we will resolve duplicates later.",
        "",
        "TASK SUMMARY:",
        "- Extract entities that appear in the FOCUS chunk only (do NOT invent facts not supported by the chunk).",
        "- You may CONSULT the CONTEXT block (previous chunks concatenated) to disambiguate or resolve pronouns.",
        "- Only list mentions present in the FOCUS chunk; if a piece of CONTEXT helped resolve a pronoun or ambiguity, include a short excerpt in `used_context_excerpt` (optional).",
        "",
        "CRITICAL INSTRUCTION FOR MECHANISMS / PROCESSES:",
        "- Always include the canonical, short **phenomenon/process label** for any mechanism/process referenced in the FOCUS chunk (e.g., 'graphitization', 'sulfidation', 'thermal fatigue'), even when the text describes it indirectly ('this type of graphitization').",
        "- For mechanisms you may return both: the short canonical name as `entity_name` and an explanatory manifestation phrase in `entity_description` or `context_phrase`.",
        "- If unsure of the canonical label, include the most likely label and mark with a moderate confidence (e.g., 0.5 - 0.7).",
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
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 800, temperature: float = 0.0) -> str:
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
    preview = txt if len(txt) <= 1000 else txt[:1000] + "\n\n...[TRUNCATED OUTPUT]"
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
chunk_ids = [f"Ch_{i:06d}" for i in range(0, 224)]



def run_entity_extraction_on_chunks(chunk_ids, prev_chunks: int = 1, save_debug: bool = False, debug_dir: str = DEFAULT_DEBUG_DIR):
    all_results = []
    for cid in chunk_ids:
        res = extract_entities_from_chunk(cid, CHUNKS_JSONL, prev_chunks=prev_chunks, model="gpt-4o", max_tokens=1000, save_debug=save_debug, debug_dir=debug_dir)
        if res:
            all_results.extend(res)
    return all_results

# Example run:
if __name__ == "__main__":
    # set save_debug=True to persist full prompt+llm output (and focus/context text) to files in DEFAULT_DEBUG_DIR
    run_entity_extraction_on_chunks(chunk_ids, prev_chunks=5, save_debug=True)


#endregion#? Entity Recognition V6 - Broader hint better prmpting
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Entity Resolution V0

#!/usr/bin/env python3
"""
entity_resolution_pipeline.py

LLM-first, cluster-driven Entity Resolution pipeline.

Input:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl

Outputs (folder created):
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/
    - entities_resolved.jsonl
    - resolve_map.json
    - resolution_history.jsonl
    - clusters_for_review.jsonl

How it works (short):
  1. Load entity mentions (id, entity_name, entity_description, chunk_id, entity_type_hint, confidence_score...)
  2. Compute embeddings for name, description, and context (if present)
  3. Build composite similarity (weighted combination)
  4. Build similarity graph with T_base and extract connected components (clusters)
  5. Auto-merge trivial clusters (very high sim)
  6. For remaining clusters, call LLM in micro-batches (<= max_batch) to decide merge/rename/keep
  7. Apply actions, log history, produce resolved entities and mapping
  8. Optional second pass to catch transitive aliases

Requires:
  transformers, torch, faiss (or faiss-cpu), openai (or openai-python compatible), tqdm, sklearn (optional but helpful)

Configure thresholds and models at the top.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import math
import time
import uuid
import itertools
from typing import List, Dict, Any, Optional

# --- Config (tune these) ---
INPUT_ENT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTest.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model (HuggingFace). Use the one you used earlier for consistency.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  #"BAAI/bge-small-en-v1.5"  # change to large if you want and have resources
USE_CUDA = True  # set False to force CPU
BATCH_EMBED = 64

# Neighbor retrieval settings
TOPN_FAST = 64   # initial retrieval size (fast)
T_BASE = 0.75    # base graph threshold for forming clusters (recall-friendly)
T_AUTO = 0.92    # auto-merge threshold (very safe)
T_HIGH = 0.95    # very high similarity (definitely same)
CROSS_ENCODER_RANGE = (0.70, 0.92)  # range where we prefer LLM rerankers (we will use cluster-LLM instead)

# Composite similarity weights (name, desc, ctx, type)
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}

# LLM settings
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be present to use LLM decisions
MAX_LLM_BATCH = 10  # maximum cluster size to feed LLM intact; larger clusters are split
MAX_PASSES = 3
MIN_MERGE_DELTA = 5

# Safety / review flags
FLAG_REVIEW_MIN_SIZE = 6  # clusters >= this size will be flagged for review
FLAG_REVIEW_SIM_THRESHOLD = 0.8  # clusters with mean pairwise sim < this flagged

# Optional: path to precomputed entity-level embeddings (if you have them)
PRECOMPUTED_EMB_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring")  # optional

# --- End config ---

# ------------- Imports that may be missing in minimal env --------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError("Missing transformers/torch. Install with `pip install torch transformers`") from e

try:
    import faiss
except Exception:
    try:
        import faiss_cpu as faiss  # sometimes named differently
    except Exception:
        raise RuntimeError("Missing faiss. Install faiss (faiss-cpu) to continue.") from None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

# sklearn optional for kmeans splitting large clusters
try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# OpenAI client compatibility
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

# -------------------- Utilities --------------------
def load_entities(path: Path) -> List[Dict[str, Any]]:
    ents = []
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            ents.append(json.loads(ln))
    print(f"[load_entities] loaded {len(ents)} entity mentions")
    return ents

# simple normalizer for surface forms
def normalize_surface(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

# cosine helper (assumes numpy arrays)
import numpy as np
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

# ---------------- Embedding helpers ----------------
print("[init] loading tokenizer and model for embeddings...")
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
model.eval()
print(f"[init] embedding model loaded on {device}")

@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = BATCH_EMBED) -> np.ndarray:
    """Return L2-normalized embeddings (float32) for list of texts"""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        if hasattr(out, "last_hidden_state"):
            token_emb = out.last_hidden_state  # (B, T, D)
        else:
            token_emb = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        token_emb = token_emb * mask
        summed = token_emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / denom
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        vecs.append(mean_emb.cpu().numpy().astype("float32"))
    if vecs:
        return np.vstack(vecs)
    return np.zeros((0, model.config.hidden_size), dtype="float32")

# ---------------- Build embeddings per field ----------------
def prepare_entity_embeddings(entities: List[Dict[str,Any]]):
    """
    Build dicts: emb_name[id] -> vector, emb_desc[id], emb_ctx[id].
    If text missing, no vector (key absent).
    """
    print("[embed] preparing name/desc/context texts")
    ids = [e["id"] for e in entities]
    name_texts = [e.get("entity_name","") or "" for e in entities]
    desc_texts = [ (e.get("entity_description") or "") for e in entities ]
    ctx_texts = []
    for e in entities:
        # build small context excerpt: prefer used_context_ids? else chunk excerpt in _raw_llm or nothing
        c = e.get("used_context_excerpt") or e.get("context_phrase") or ""
        ctx_texts.append(c[:512])  # cap length

    # compute embeddings in batches
    emb_name = embed_texts(name_texts)
    emb_desc = embed_texts(desc_texts)
    emb_ctx = embed_texts(ctx_texts)

    # convert to dict keyed by id, drop-empty heuristics (we keep all but could drop zeros)
    emb_name_map = { ids[i]: emb_name[i] for i in range(len(ids)) }
    emb_desc_map = { ids[i]: emb_desc[i] for i in range(len(ids)) }
    emb_ctx_map  = { ids[i]: emb_ctx[i]  for i in range(len(ids)) }
    print(f"[embed] built embeddings: name:{len(emb_name_map)} desc:{len(emb_desc_map)} ctx:{len(emb_ctx_map)}")
    return emb_name_map, emb_desc_map, emb_ctx_map

# ------------------ Composite similarity ------------------
def composite_sim(e1: Dict, e2: Dict, emb_name, emb_desc, emb_ctx, weights=WEIGHTS):
    sims = []
    ws = []
    # name
    v1 = emb_name.get(e1["id"])
    v2 = emb_name.get(e2["id"])
    if v1 is not None and v2 is not None:
        sims.append(cosine(v1, v2))
        ws.append(weights["name"])
    # desc
    v1 = emb_desc.get(e1["id"])
    v2 = emb_desc.get(e2["id"])
    if v1 is not None and v2 is not None and (v1.any() or v2.any()):
        sims.append(cosine(v1, v2))
        ws.append(weights["desc"])
    # ctx
    v1 = emb_ctx.get(e1["id"])
    v2 = emb_ctx.get(e2["id"])
    if v1 is not None and v2 is not None and (v1.any() or v2.any()):
        sims.append(cosine(v1, v2))
        ws.append(weights["ctx"])
    # type hint (simple equality)
    t1 = (e1.get("entity_type_hint") or "").strip().lower()
    t2 = (e2.get("entity_type_hint") or "").strip().lower()
    if t1 and t2:
        sims.append(1.0 if t1 == t2 else 0.0)
        ws.append(weights["type"])

    if not sims:
        return 0.0
    total_w = sum(ws)
    if total_w <= 0:
        return sum(sims) / len(sims)
    weighted = sum(s * w for s, w in zip(sims, ws)) / total_w
    return float(weighted)

# ------------------ FAISS fast neighbor retrieval ------------------
def build_faiss_index(emb_map: Dict[str, np.ndarray]):
    ids = list(emb_map.keys())
    vecs = np.vstack([emb_map[i] for i in ids]).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, ids, vecs

def neighbors_topn(index, ids_list, vecs, query_vec, topn=TOPN_FAST):
    q = query_vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, topn)
    res = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(ids_list):
            continue
        res.append((ids_list[idx], float(dist)))
    return res

# ------------------ Graph & clusters ------------------
def build_similarity_graph(entities, emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE):
    print("[graph] building candidate edges with base threshold", t_base)
    n = len(entities)
    id_map = {e["id"]: e for e in entities}
    edges = defaultdict(list)
    # iterate over all entities and query fast index on name embedding (fallback to desc)
    for e in tqdm(entities, desc="neighbors"):
        qid = e["id"]
        # prefer name embedding for quick retrieval; fallback to desc or ctx
        # qvec = emb_name.get(qid) or emb_desc.get(qid) or emb_ctx.get(qid)
        qvec = emb_name.get(qid)
        if qvec is None:
            qvec = emb_desc.get(qid)
        if qvec is None:
            qvec = emb_ctx.get(qid)
        if qvec is None:
            continue
        nbrs = neighbors_topn(fast_index, fast_ids, fast_vecs, qvec, topn=TOPN_FAST)
        for nid, fast_score in nbrs:
            if nid == qid:
                continue
            # compute composite exact sim
            s = composite_sim(e, id_map[nid], emb_name, emb_desc, emb_ctx)
            if s >= t_base:
                edges[qid].append((nid, s))
    # build connected components via simple DFS
    visited = set()
    clusters = []
    for e in entities:
        eid = e["id"]
        if eid in visited:
            continue
        stack = [eid]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for (nbr, _) in edges.get(cur, []):
                if nbr not in visited:
                    stack.append(nbr)
            # also consider incoming edges (make graph undirected)
            for src, nbrs in edges.items():
                for (n2, _) in nbrs:
                    if n2 == cur and src not in visited:
                        stack.append(src)
        clusters.append(sorted(list(comp)))
    print(f"[graph] built {len(clusters)} candidate clusters (including singletons)")
    return clusters, edges

# ------------------ Union-Find ------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, a):
        if a not in self.parent:
            self.parent[a] = a
            return a
        if self.parent[a] == a:
            return a
        self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra
    def groups(self):
        d = {}
        for x in list(self.parent.keys()):
            r = self.find(x)
            d.setdefault(r, []).append(x)
        return list(d.values())

# ------------------ LLM cluster prompt builder & caller ------------------
def build_cluster_prompt(cluster_entities: List[Dict], entities_by_id: Dict[str, Dict], examples: Optional[List[Dict]] = None):
    """
    Build strict JSON-returning prompt for cluster micro-batch.
    The LLM must return ONLY a JSON array of actions.
    """
    header = [
        "You are an assistant whose sole job is to decide whether entity mentions refer to the same real-world entity.",
        "Important: Do NOT create classes or types. Only decide merges/renames/keeps for the entity mentions provided.",
        "Return ONLY a JSON array. Each element is an object with one of these action types:",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...'}",
        " - rename_entity: {action:'rename_entity', entity_id:'...', new_name:'...', new_description:'...'}",
        " - keep_entity: {action:'keep_entity', entity_id:'...'}",
        "Be concise in descriptions (one sentence). If unsure, prefer keep_entity rather than incorrect merge.",
        ""
    ]
    if examples:
        header.append("Examples (follow these shapes exactly):")
        header.append(json.dumps(examples, ensure_ascii=False, indent=2))
        header.append("")

    header.append("INPUT: the following mentions (id, surface, short_desc, chunk_excerpt (optional), type_hint). Decide merges/renames/keeps.")
    header_text = "\n".join(header)

    items = []
    for e in cluster_entities:
        ent = entities_by_id[e]
        items.append({
            "id": ent.get("id"),
            "surface": ent.get("entity_name"),
            "short_desc": (ent.get("entity_description") or "")[:200],
            "context_excerpt": (ent.get("used_context_excerpt") or ent.get("context_phrase") or "")[:300],
            "type_hint": ent.get("entity_type_hint")
        })
    prompt = header_text + "\n\n" + json.dumps({"mentions": items}, ensure_ascii=False, indent=2)
    prompt += "\n\nReturn only a JSON array of actions. No commentary."
    return prompt

def call_llm(prompt: str, model=OPENAI_MODEL, max_tokens=600, temperature=0.0):
    if not OPENAI_API_KEY or not OPENAI_CLIENT_AVAILABLE:
        print("[LLM] OpenAI key or client not available. Skipping real LLM call.")
        return None
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        txt = resp.choices[0].message.content
        return txt
    except Exception as e:
        print("[LLM] call error:", e)
        return None

def safe_parse_llm_json(txt: str):
    if not txt:
        return []
    t = txt.strip()
    # unwrap code fences
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json", "", 1).strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        # try tolerant fixes
        try:
            parsed = json.loads(t.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            print("[LLM] parse failed:", e)
            print("Raw:", t[:1000])
            return []
    return []

# ------------------ Apply actions ------------------
def apply_actions(actions: List[Dict], entities_by_id: Dict[str, Dict], uf: UnionFind, resolved_records: Dict[str, Dict], history: List[Dict]):
    """
    actions: list of action dicts from LLM.
    We apply them by unioning merged ids in union-find and by recording renames/keeps in history.
    resolved_records is not built here fully; we just mark in resolve_map via uf.
    """
    for act in actions:
        action = act.get("action")
        if action == "merge_entities":
            mids = act.get("merged_ids") or []
            canonical_name = act.get("canonical_name") or act.get("canonical')", "")
            for i in range(1, len(mids)):
                uf.union(mids[0], mids[i])
            # record history
            history.append({
                "ts": time.time(),
                "action": "merge_entities",
                "merged_ids": mids,
                "canonical_name": canonical_name,
                "new_description": act.get("new_description")
            })
        elif action == "rename_entity":
            eid = act.get("entity_id")
            new_name = act.get("new_name")
            history.append({
                "ts": time.time(),
                "action": "rename_entity",
                "entity_id": eid,
                "new_name": new_name,
                "new_description": act.get("new_description")
            })
            # we can record rename in resolved_records later when constructing canonical node
            # for now, attach to entity
            if eid in entities_by_id:
                entities_by_id[eid]["__rename_suggested"] = {"new_name": new_name, "new_description": act.get("new_description")}
        elif action == "keep_entity":
            eid = act.get("entity_id")
            history.append({"ts": time.time(), "action": "keep_entity", "entity_id": eid})
        else:
            # unknown action: ignore but log
            history.append({"ts": time.time(), "action": "unknown", "payload": act})

# ------------------ Main pipeline ------------------
def main():
    entities = load_entities(INPUT_ENT_PATH)
    entities_by_id = { e["id"]: e for e in entities }

    # Prepare embeddings
    emb_name, emb_desc, emb_ctx = prepare_entity_embeddings(entities)

    # Build fast index on name embeddings (fallback to desc if name missing)
    # Combine name+desc as a fallback vector for indexing: simple average
    combined_emb_map = {}
    for eid in emb_name:
        combined_emb_map[eid] = emb_name[eid]
    for eid in emb_desc:
        if eid not in combined_emb_map:
            combined_emb_map[eid] = emb_desc[eid]
        else:
            combined_emb_map[eid] = (combined_emb_map[eid] + emb_desc[eid]) / 2.0

    # if after this some entities still missing, use ctx
    for eid in emb_ctx:
        if eid not in combined_emb_map:
            combined_emb_map[eid] = emb_ctx[eid]

    if not combined_emb_map:
        raise RuntimeError("No embeddings available to build fast index; cannot proceed")

    fast_index, fast_ids, fast_vecs = build_faiss_index(combined_emb_map)

    # Build similarity graph and clusters
    clusters, edges = build_similarity_graph(entities, emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE)

    # classify clusters: trivial singletons, tiny-auto-merge, need-LLM
    singleton_count = sum(1 for c in clusters if len(c) == 1)
    print(f"[clusters] total clusters: {len(clusters)}, singletons: {singleton_count}")

    # Pre-prepare union-find and history
    uf = UnionFind()
    history = []
    # if auto-merge: identify cluster mean similarity
    def mean_pairwise_sim(cluster_list):
        ids = cluster_list
        if len(ids) <= 1:
            return 1.0
        sims = []
        for a, b in itertools.combinations(ids, 2):
            sim = composite_sim(entities_by_id[a], entities_by_id[b], emb_name, emb_desc, emb_ctx)
            sims.append(sim)
        return float(sum(sims) / len(sims)) if sims else 0.0

    clusters_for_llm = []
    clusters_for_review = []

    for cl in clusters:
        if len(cl) == 1:
            # singleton: keep as-is (no merge)
            eid = cl[0]
            history.append({"ts": time.time(), "action": "kept_singleton", "entity_id": eid})
            uf.find(eid)  # ensure presence in union-find
            continue
        mps = mean_pairwise_sim(cl)
        if len(cl) <= 3 and mps >= T_AUTO:
            # safe auto-merge: union all
            for i in range(1, len(cl)):
                uf.union(cl[0], cl[i])
            history.append({"ts": time.time(), "action": "auto_merge", "member_ids": cl, "mean_sim": mps})
            continue
        # else candidate for LLM
        clusters_for_llm.append(cl)
        # flag large or low-mean for review later
        if len(cl) >= FLAG_REVIEW_MIN_SIZE or mps < FLAG_REVIEW_SIM_THRESHOLD:
            clusters_for_review.append({"cluster": cl, "mean_sim": mps})

    print(f"[plan] clusters_for_llm: {len(clusters_for_llm)}, clusters flagged_for_review: {len(clusters_for_review)}")

    # LLM handling: for each cluster, split if needed, call LLM, apply actions
    # Helper to split cluster into batches <= MAX_LLM_BATCH using KMeans if available
    def split_cluster(cluster_ids, max_batch=MAX_LLM_BATCH):
        if len(cluster_ids) <= max_batch:
            return [cluster_ids]
        # try kmeans on combined_emb_map vectors
        if _SKLEARN_AVAILABLE:
            k = math.ceil(len(cluster_ids) / max_batch)
            X = np.vstack([combined_emb_map[cid] for cid in cluster_ids])
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            parts = defaultdict(list)
            for cid, lbl in zip(cluster_ids, km.labels_):
                parts[int(lbl)].append(cid)
            out = [parts[i] for i in sorted(parts.keys())]
            # ensure no part exceeds max_batch; if so, chunk sequentially
            final = []
            for p in out:
                if len(p) <= max_batch:
                    final.append(p)
                else:
                    for i in range(0, len(p), max_batch):
                        final.append(p[i:i+max_batch])
            return final
        else:
            # fallback sequential chunking
            out = []
            for i in range(0, len(cluster_ids), max_batch):
                out.append(cluster_ids[i:i+max_batch])
            return out

    # examples for LLM prompt
    examples = [
        {"action":"merge_entities","merged_ids":["En_A","En_B"],"canonical_name":"graphitization","new_description":"The physical process of graphite formation in steel that reduces mechanical strength."},
        {"action":"rename_entity","entity_id":"En_C","new_name":"MotifX_Algorithm","new_description":"Algorithm for detecting recurring motifs in graphs."},
        {"action":"keep_entity","entity_id":"En_D"}
    ]

    # iterate clusters_for_llm
    llm_calls = 0
    for cl in tqdm(clusters_for_llm, desc="LLM clusters"):
        subparts = split_cluster(cl, max_batch=MAX_LLM_BATCH)
        for sub in subparts:
            # skip already-resolved ones (could happen due to unions from previous)
            sub_active = [s for s in sub if uf.find(s) == s]  # only roots unmerged
            if not sub_active:
                continue
            cluster_prompt = build_cluster_prompt(sub_active, entities_by_id, examples=examples)
            llm_calls += 1
            if OPENAI_API_KEY and OPENAI_CLIENT_AVAILABLE:
                raw = call_llm(cluster_prompt)
                if raw is None:
                    # if call failed, log and add to review
                    clusters_for_review.append({"cluster": sub_active, "reason": "llm_error"})
                    continue
                actions = safe_parse_llm_json(raw)
                if not actions:
                    # parser failed or empty -> flag review
                    clusters_for_review.append({"cluster": sub_active, "reason": "llm_no_json"})
                    continue
                # apply actions
                apply_actions(actions, entities_by_id, uf, {}, history)
            else:
                # no LLM available: flag for review (we wrote cluster content)
                clusters_for_review.append({"cluster": sub_active, "reason": "no_llm_key"})
    print(f"[LLM] performed approx {llm_calls} LLM micro-batch calls (if key present).")

    # After LLM pass(s), optional second pass: re-embed canonical nodes and re-run clustering if desired
    # For simplicity we will do one more quick pass: compute groups from union-find and create canonical nodes
    all_entities = entities_by_id
    # ensure every entity appears in uf
    for eid in list(all_entities.keys()):
        uf.find(eid)

    groups = {}
    for eid in all_entities.keys():
        root = uf.find(eid)
        groups.setdefault(root, []).append(eid)

    print(f"[result] union-find groups produced {len(groups)} resolved groups")

    # Build resolved records
    resolved_records = []
    resolve_map = {}
    for root, members in groups.items():
        # choose canonical name: prefer recommended rename if any; else most common surface normalized; else longest
        candidate_names = []
        for m in members:
            r = entities_by_id.get(m, {})
            rn = r.get("__rename_suggested", {}).get("new_name")
            if rn:
                candidate_names.append(rn)
        if candidate_names:
            canonical_name = Counter(candidate_names).most_common(1)[0][0]
        else:
            surfaces = [entities_by_id[m].get("entity_name") or "" for m in members]
            normalized = [normalize_surface(s) for s in surfaces]
            if normalized:
                canonical_name = Counter(normalized).most_common(1)[0][0]
            else:
                canonical_name = f"Resolved_{root}"
        # description: collect any suggested descriptions in history or from members
        descs = []
        for m in members:
            s = entities_by_id[m].get("__rename_suggested", {}).get("new_description")
            if s:
                descs.append(s)
        if not descs:
            # fallback to concatenated short descriptions from members (first 200 chars)
            descs = [ (entities_by_id[m].get("entity_description") or "")[:200] for m in members if entities_by_id[m].get("entity_description")]
        description = descs[0] if descs else f"Resolved entity combining {len(members)} mentions."
        new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
        resolved = {
            "id_final": new_id,
            "label": canonical_name,
            "aliases": list({ entities_by_id[m].get("entity_name") for m in members if entities_by_id[m].get("entity_name") != canonical_name }),
            "description": description,
            "members": members,
            "flag": "resolved_entity"
        }
        resolved_records.append(resolved)
        for m in members:
            resolve_map[m] = new_id

    # Save outputs
    ENT_OUT = OUT_DIR / "entities_resolved.jsonl"
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for r in resolved_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    MAP_OUT = OUT_DIR / "resolve_map.json"
    with open(MAP_OUT, "w", encoding="utf-8") as fh:
        json.dump(resolve_map, fh, indent=2, ensure_ascii=False)
    HISTORY_OUT = OUT_DIR / "resolution_history.jsonl"
    with open(HISTORY_OUT, "w", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    REVIEW_OUT = OUT_DIR / "clusters_for_review.jsonl"
    with open(REVIEW_OUT, "w", encoding="utf-8") as fh:
        for c in clusters_for_review:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("[done] results written to:", OUT_DIR)
    print(" - entities_resolved:", ENT_OUT)
    print(" - resolve_map:", MAP_OUT)
    print(" - history:", HISTORY_OUT)
    print(" - clusters_for_review:", REVIEW_OUT)
    print("Summary:")
    print(f"  original_mentions: {len(entities)}")
    print(f"  resolved_entities (canonical): {len(resolved_records)}")
    print(f"  clusters_for_review: {len(clusters_for_review)}")

if __name__ == "__main__":
    main()


#endregion#? Entity Resolution V0
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Entity Resolution V1



#!/usr/bin/env python3
"""
entity_resolution_pipeline_v1_parsefix.py

Same pipeline as before but improved LLM parsing robustness, debug logging,
and a single retry attempt when the LLM returns unparsable JSON.

Keep prior behavior (MIN_SIM, auto-merge, LLM decisions, iterative passes).
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import math
import time
import uuid
import itertools
import re
from typing import List, Dict, Any, Optional

# ---------------- Config (tune these) ----------------
INPUT_ENT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
USE_CUDA = True
BATCH_EMBED = 64

# Neighbor retrieval / thresholds
TOPN_FAST = 64
T_BASE = 0.75
T_AUTO = 0.94
MIN_SIM = 0.40

WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}

OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_LLM_BATCH = 10
MAX_PASSES = 4

# Debug and retry
VERBOSE = True
DEBUG_DIR = OUT_DIR / "llm_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
LLM_PARSE_RETRY = True  # perform 1 retry if parse fails

# ---------------- Dependencies ----------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError("Missing transformers/torch. Install with `pip install torch transformers`") from e

try:
    import faiss
except Exception:
    try:
        import faiss_cpu as faiss
    except Exception:
        raise RuntimeError("Missing faiss. Install faiss (faiss-cpu) to continue.") from None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# OpenAI client attempt
_OPENAI_CLIENT = None
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_CLIENT = ("openai_client", OpenAIClient)
except Exception:
    try:
        import openai
        _OPENAI_CLIENT = ("openai_pkg", openai)
    except Exception:
        _OPENAI_CLIENT = None

# ---------------- Utilities ----------------
def load_entities(path: Path) -> List[Dict[str, Any]]:
    ents = []
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            ents.append(json.loads(ln))
    if VERBOSE:
        print(f"[load_entities] loaded {len(ents)} entity mentions from {path}")
    return ents

def normalize_surface(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

import numpy as np
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

# ---------------- Embedding ----------------
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
if VERBOSE:
    print(f"[init] loading embedding model {EMBED_MODEL} on {device}")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
model.eval()

@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = BATCH_EMBED) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        token_emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        token_emb = token_emb * mask
        summed = token_emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / denom
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        vecs.append(mean_emb.cpu().numpy().astype("float32"))
    if vecs:
        return np.vstack(vecs)
    return np.zeros((0, model.config.hidden_size), dtype="float32")

def prepare_entity_embeddings(entities: List[Dict[str,Any]]):
    if VERBOSE:
        print("[embed] preparing name/desc/context texts for embedding")
    ids = [e["id"] for e in entities]
    name_texts = [e.get("entity_name","") or "" for e in entities]
    desc_texts = [ (e.get("entity_description") or "") for e in entities ]
    ctx_texts = []
    for e in entities:
        c = e.get("used_context_excerpt") or e.get("context_phrase") or ""
        ctx_texts.append(c[:512])
    emb_name = embed_texts(name_texts)
    emb_desc = embed_texts(desc_texts)
    emb_ctx = embed_texts(ctx_texts)
    emb_name_map = { ids[i]: emb_name[i] for i in range(len(ids)) }
    emb_desc_map = { ids[i]: emb_desc[i] for i in range(len(ids)) }
    emb_ctx_map  = { ids[i]: emb_ctx[i]  for i in range(len(ids)) }
    if VERBOSE:
        print(f"[embed] built embeddings: name:{len(emb_name_map)} desc:{len(emb_desc_map)} ctx:{len(emb_ctx_map)}")
    return emb_name_map, emb_desc_map, emb_ctx_map

# ---------------- Composite sim / FAISS / graph ----------------
def composite_sim(e1: Dict, e2: Dict, emb_name, emb_desc, emb_ctx, weights=WEIGHTS):
    sims = []
    ws = []
    v1 = emb_name.get(e1["id"]); v2 = emb_name.get(e2["id"])
    if v1 is not None and v2 is not None:
        sims.append(cosine(v1, v2)); ws.append(weights["name"])
    v1 = emb_desc.get(e1["id"]); v2 = emb_desc.get(e2["id"])
    if v1 is not None and v2 is not None and (np.any(v1) or np.any(v2)):
        sims.append(cosine(v1, v2)); ws.append(weights["desc"])
    v1 = emb_ctx.get(e1["id"]); v2 = emb_ctx.get(e2["id"])
    if v1 is not None and v2 is not None and (np.any(v1) or np.any(v2)):
        sims.append(cosine(v1, v2)); ws.append(weights["ctx"])
    t1 = (e1.get("entity_type_hint") or "").strip().lower()
    t2 = (e2.get("entity_type_hint") or "").strip().lower()
    if t1 and t2:
        sims.append(1.0 if t1 == t2 else 0.0); ws.append(weights["type"])
    if not sims:
        return 0.0
    total_w = sum(ws)
    if total_w <= 0:
        return sum(sims) / len(sims)
    weighted = sum(s * w for s, w in zip(sims, ws)) / total_w
    return float(weighted)

def build_faiss_index(emb_map: Dict[str, np.ndarray]):
    ids = list(emb_map.keys())
    vecs = np.vstack([emb_map[i] for i in ids]).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, ids, vecs

def neighbors_topn(index, ids_list, vecs, query_vec, topn=TOPN_FAST):
    q = query_vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, topn)
    res = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(ids_list):
            continue
        res.append((ids_list[idx], float(dist)))
    return res

def build_similarity_graph(entities, emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE):
    if VERBOSE:
        print("[graph] building candidate edges with base threshold", t_base)
    id_map = {e["id"]: e for e in entities}
    adjacency = defaultdict(dict)
    for e in tqdm(entities, desc="neighbors"):
        qid = e["id"]
        # prefer name then desc then ctx: use explicit membership to avoid KeyError
        if qid in emb_name:
            qvec = emb_name[qid]
        elif qid in emb_desc:
            qvec = emb_desc[qid]
        elif qid in emb_ctx:
            qvec = emb_ctx[qid]
        else:
            qvec = None
        if qvec is None:
            continue
        nbrs = neighbors_topn(fast_index, fast_ids, fast_vecs, qvec, topn=TOPN_FAST)
        for nid, fast_score in nbrs:
            if nid == qid:
                continue
            s = composite_sim(e, id_map[nid], emb_name, emb_desc, emb_ctx)
            if s >= t_base:
                adjacency[qid][nid] = s
                adjacency[nid][qid] = s
    visited = set()
    clusters = []
    for eid in id_map.keys():
        if eid in visited:
            continue
        stack = [eid]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nbr in adjacency.get(cur, {}).keys():
                if nbr not in visited:
                    stack.append(nbr)
        clusters.append(sorted(comp))
    if VERBOSE:
        print(f"[graph] built {len(clusters)} candidate clusters (including singletons)")
    return clusters, adjacency

# ---------------- Union-Find ----------------
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, a):
        if a not in self.parent:
            self.parent[a] = a
            return a
        if self.parent[a] == a:
            return a
        self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra
    def groups(self):
        d = {}
        for x in list(self.parent.keys()):
            r = self.find(x)
            d.setdefault(r, []).append(x)
        return list(d.values())

# ---------------- LLM prompt + call + parsing ----------------
def build_cluster_prompt(cluster_entity_ids: List[str], entities_by_id: Dict[str, Dict], examples: Optional[List[Dict]]=None):
    header = [
        "You are an assistant whose job is to decide whether entity mentions refer to the same real-world entity.",
        "Return ONLY a JSON array (no commentary). Each element must be an action object.",
        "Action types (choose one per object):",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - rename_entity: {action:'rename_entity', entity_id:'...', new_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - keep_entity: {action:'keep_entity', entity_id:'...', confidence:0.0-1.0}",
        "Important: include a 'confidence' float (0.0-1.0) for every action.",
        "If unsure, prefer keep_entity with confidence < 0.9.",
        ""
    ]
    # two concise examples to encourage correct shape
    examples_block = [
        {"action":"merge_entities","merged_ids":["En_A","En_B"],"canonical_name":"graphitization","new_description":"Formation of graphite in steel reducing strength.","confidence":0.95},
        {"action":"rename_entity","entity_id":"En_C","new_name":"Heat_Fatigue","new_description":"Thermal fatigue cause in welds.", "confidence":0.85},
        {"action":"keep_entity","entity_id":"En_D","confidence":0.40}
    ]
    if examples is None:
        header.append("EXAMPLES:")
        header.append(json.dumps(examples_block, ensure_ascii=False, indent=2))
        header.append("")
    else:
        header.append("EXAMPLES:")
        header.append(json.dumps(examples, ensure_ascii=False, indent=2))
        header.append("")
    header_text = "\n".join(header)
    items = []
    for eid in cluster_entity_ids:
        ent = entities_by_id[eid]
        items.append({
            "id": ent.get("id"),
            "surface": ent.get("entity_name"),
            "short_desc": (ent.get("entity_description") or "")[:200],
            "context_excerpt": (ent.get("used_context_excerpt") or ent.get("context_phrase") or "")[:300],
            "type_hint": ent.get("entity_type_hint")
        })
    prompt = header_text + "\n\n" + json.dumps({"mentions": items}, ensure_ascii=False, indent=2)
    prompt += "\n\nReturn only a JSON array of actions."
    return prompt

def call_llm(prompt: str, model=OPENAI_MODEL, max_tokens=600, temperature=0.0):
    if not OPENAI_API_KEY or _OPENAI_CLIENT is None:
        if VERBOSE:
            print("[LLM] OpenAI key or client not available. Skipping LLM call.")
        return None
    try:
        kind, client_pkg = _OPENAI_CLIENT
        if kind == "openai_client":
            client = client_pkg(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            txt = resp.choices[0].message.content
            return txt
        else:
            client_pkg.api_key = OPENAI_API_KEY
            resp = client_pkg.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            txt = resp.choices[0].message.content
            return txt
    except Exception as e:
        print("[LLM] call error:", e)
        return None

def extract_first_json_array(txt: str):
    """
    Robust extraction attempt:
     1) find first [...] block and parse
     2) if fails, scan for balanced {...} objects and parse them individually; return list of parsed objects
     3) try single-quote tolerant load
    """
    if not txt:
        return []
    t = txt.strip()
    # remove leading/trailing triple backticks
    if t.startswith("```"):
        t = t.strip("`").strip()
    # 1) try to find the first JSON array block
    match = re.search(r'(\[\s*(?:[\s\S]*?)\s*\])', t)
    if match:
        candidate = match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            # try single-quote tolerant
            try:
                parsed = json.loads(candidate.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    # 2) fallback: attempt to extract balanced JSON objects {...}
    objs = []
    stack = []
    start_idx = None
    for i, ch in enumerate(t):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    piece = t[start_idx:i+1]
                    try:
                        obj = json.loads(piece)
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except Exception:
                        # try single-quote tolerant
                        try:
                            obj = json.loads(piece.replace("'", '"'))
                            if isinstance(obj, dict):
                                objs.append(obj)
                        except Exception:
                            pass
                    start_idx = None
    if objs:
        return objs
    # 3) final fallback: try to json.load entire text
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        parsed = json.loads(t.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []

# ---------------- apply actions (robust) ----------------
def apply_actions(actions: List[Dict], entities_by_id: Dict[str, Dict], uf: UnionFind, history: List[Dict]):
    changed = False
    for act in actions:
        if not isinstance(act, dict):
            history.append({"ts": time.time(), "action":"invalid_action_format", "payload": act})
            continue
        action = act.get("action")
        conf = float(act.get("confidence")) if act.get("confidence") is not None else None
        if action == "merge_entities":
            mids = act.get("merged_ids") or []
            if len(mids) >= 2:
                base = mids[0]
                for other in mids[1:]:
                    uf.union(base, other)
                changed = True
            canonical_name = act.get("canonical_name") or act.get("canonical") or ""
            history.append({"ts": time.time(), "action":"merge_entities", "merged_ids": mids, "canonical_name": canonical_name, "new_description": act.get("new_description"), "confidence": conf})
        elif action == "rename_entity":
            eid = act.get("entity_id")
            new_name = act.get("new_name")
            if eid in entities_by_id:
                entities_by_id[eid]["__rename_suggested"] = {"new_name": new_name, "new_description": act.get("new_description"), "confidence": conf}
            history.append({"ts": time.time(), "action":"rename_entity", "entity_id": eid, "new_name": new_name, "new_description": act.get("new_description"), "confidence": conf})
        elif action == "keep_entity":
            eid = act.get("entity_id")
            history.append({"ts": time.time(), "action":"keep_entity", "entity_id": eid, "confidence": conf})
        else:
            history.append({"ts": time.time(), "action":"unknown", "payload": act})
    return changed

# ---------------- cluster splitting ----------------
def split_cluster(cluster_ids: List[str], max_batch=MAX_LLM_BATCH):
    if len(cluster_ids) <= max_batch:
        return [cluster_ids]
    out = []
    for i in range(0, len(cluster_ids), max_batch):
        out.append(cluster_ids[i:i+max_batch])
    return out

def mean_pairwise_sim(cluster_list: List[str], entities_by_id: Dict[str, Dict], emb_name, emb_desc, emb_ctx):
    if len(cluster_list) <= 1:
        return 1.0
    sims = []
    for a, b in itertools.combinations(cluster_list, 2):
        sims.append(composite_sim(entities_by_id[a], entities_by_id[b], emb_name, emb_desc, emb_ctx))
    return float(sum(sims)/len(sims)) if sims else 0.0

# ---------------- helper: save debug file ----------------
def save_llm_debug_file(cluster_root_tag: str, pass_no: int, sub_index: int, prompt: str, raw: str, raw_retry: Optional[str]=None):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fname = f"debug_pass{pass_no}_cluster{cluster_root_tag}_sub{sub_index}_{ts}_{uuid.uuid4().hex[:6]}.json"
    p = DEBUG_DIR / fname
    payload = {"pass": pass_no, "cluster_tag": cluster_root_tag, "sub_index": sub_index, "prompt": prompt, "raw": raw, "raw_retry": raw_retry, "timestamp": ts}
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return str(p)

# ---------------- Main pipeline (with retry on parse failure) ----------------
def main():
    entities = load_entities(INPUT_ENT_PATH)
    entities_by_id = { e["id"]: e for e in entities }

    emb_name, emb_desc, emb_ctx = prepare_entity_embeddings(entities)

    combined_emb_map = {}
    for eid, v in emb_name.items():
        combined_emb_map[eid] = v.copy()
    for eid, v in emb_desc.items():
        if eid not in combined_emb_map:
            combined_emb_map[eid] = v.copy()
        else:
            combined_emb_map[eid] = (combined_emb_map[eid] + v) / 2.0
    for eid, v in emb_ctx.items():
        if eid not in combined_emb_map:
            combined_emb_map[eid] = v.copy()

    if not combined_emb_map:
        raise RuntimeError("No embeddings available to build fast index; cannot proceed")
    fast_index, fast_ids, fast_vecs = build_faiss_index(combined_emb_map)

    uf = UnionFind()
    history: List[Dict] = []
    for eid in entities_by_id.keys():
        uf.find(eid)

    pass_no = 0
    total_llm_calls = 0
    changed_in_pass = True

    while pass_no < MAX_PASSES and changed_in_pass:
        pass_no += 1
        if VERBOSE:
            print(f"\n=== PASS {pass_no} ===")
        changed_in_pass = False

        clusters, adjacency = build_similarity_graph(list(entities_by_id.values()), emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE)

        clusters_to_process = []
        for cl in clusters:
            mps = mean_pairwise_sim(cl, entities_by_id, emb_name, emb_desc, emb_ctx)
            if len(cl) == 1:
                eid = cl[0]
                history.append({"ts": time.time(), "action":"kept_singleton", "entity_id": eid, "mean_sim": mps, "processed_by":"none", "confidence": mps})
                continue
            if mps < MIN_SIM:
                history.append({"ts": time.time(), "action":"left_unchanged_low_sim", "members": cl, "mean_sim": mps, "processed_by":"none", "confidence": mps})
                continue
            clusters_to_process.append((cl, mps))

        if VERBOSE:
            print(f"[pass {pass_no}] clusters_to_process: {len(clusters_to_process)} (min_sim={MIN_SIM}, auto_threshold={T_AUTO})")

        for cl, mps in tqdm(clusters_to_process, desc=f"processing clusters (pass {pass_no})"):
            roots = set(uf.find(x) for x in cl)
            if len(roots) == 1:
                continue
            if mps >= T_AUTO:
                for i in range(1, len(cl)):
                    uf.union(cl[0], cl[i])
                history.append({"ts": time.time(), "action":"auto_merge", "members": cl, "mean_sim": mps, "processed_by": "auto", "confidence": 1.0})
                changed_in_pass = True
                if VERBOSE:
                    print(f"[auto_merge] merged cluster size {len(cl)} mean_sim={mps:.3f}")
                continue

            subparts = split_cluster(cl, max_batch=MAX_LLM_BATCH)
            for idx, sub in enumerate(subparts):
                sub_active = [s for s in sub if uf.find(s) == s]
                if not sub_active:
                    continue
                cluster_prompt = build_cluster_prompt(sub_active, entities_by_id, examples=None)
                total_llm_calls += 1
                raw = call_llm(cluster_prompt)
                raw_retry = None
                actions = extract_first_json_array(raw) if raw else []
                if not actions and raw and LLM_PARSE_RETRY:
                    # retry once with corrective instruction
                    retry_prompt = (
                        "You returned an output that is not valid JSON. "
                        "Please return ONLY a single JSON array (no commentary) with the actions as specified. "
                        "Here is your previous output (do not include it in the response):\n\n"
                        "----PREVIOUS OUTPUT START----\n" + raw[:4000] + "\n----PREVIOUS OUTPUT END----\n\n"
                        "Return only a corrected JSON array now."
                    )
                    raw_retry = call_llm(retry_prompt)
                    actions = extract_first_json_array(raw_retry) if raw_retry else []
                    # save debug including both raw and retry
                    dbg_path = save_llm_debug_file(cluster_root_tag=cl[0], pass_no=pass_no, sub_index=idx+1, prompt=cluster_prompt, raw=raw or "", raw_retry=raw_retry or "")
                    if VERBOSE:
                        print(f"[LLM parse retry] saved debug file: {dbg_path}")
                elif not actions and raw:
                    dbg_path = save_llm_debug_file(cluster_root_tag=cl[0], pass_no=pass_no, sub_index=idx+1, prompt=cluster_prompt, raw=raw or "", raw_retry=None)
                    if VERBOSE:
                        print(f"[LLM parse fail] saved debug file: {dbg_path}")

                if not actions:
                    # parser failed or LLM unavailable -> fallback keeps with low confidence
                    for s in sub_active:
                        history.append({"ts": time.time(), "action":"llm_parse_failed_fallback_keep", "entity_id": s, "processed_by": "fallback", "confidence": 0.25})
                    if VERBOSE:
                        print("[LLM parse] parse failed; kept sub-batch conservatively")
                    continue

                applied = apply_actions(actions, entities_by_id, uf, history)
                if applied:
                    changed_in_pass = True
                if VERBOSE:
                    print(f"[LLM] processed sub-batch {idx+1}/{len(subparts)} for cluster(size={len(cl)}) -> actions: {len(actions)}")
        if VERBOSE:
            print(f"[pass {pass_no}] finished. changed_in_pass={changed_in_pass}")

    # assemble groups and resolved records (same as prior code)
    groups = {}
    for eid in entities_by_id.keys():
        root = uf.find(eid)
        groups.setdefault(root, []).append(eid)

    resolved_records = []
    resolve_map = {}
    clusters_processed = []

    for root, members in groups.items():
        candidate_names = []
        candidate_name_conf = []
        for m in members:
            rs = entities_by_id[m].get("__rename_suggested")
            if rs and rs.get("new_name"):
                candidate_names.append(rs.get("new_name"))
                candidate_name_conf.append(rs.get("confidence") or 0.0)
        if candidate_names:
            if candidate_name_conf:
                idx = int(np.argmax(candidate_name_conf))
                canonical_name = candidate_names[idx]
            else:
                canonical_name = Counter(candidate_names).most_common(1)[0][0]
        else:
            surfaces = [entities_by_id[m].get("entity_name") or "" for m in members]
            normalized = [normalize_surface(s) for s in surfaces]
            if normalized:
                canonical_name = Counter(normalized).most_common(1)[0][0]
            else:
                canonical_name = f"Resolved_{root}"

        descs = []
        desc_confs = []
        for m in members:
            rs = entities_by_id[m].get("__rename_suggested")
            if rs and rs.get("new_description"):
                descs.append(rs.get("new_description"))
                desc_confs.append(rs.get("confidence") or 0.0)
        if descs:
            idx = int(np.argmax(desc_confs)) if desc_confs else 0
            description = descs[idx]
        else:
            found = None
            for m in members:
                dd = entities_by_id[m].get("entity_description")
                if dd:
                    found = dd
                    break
            description = found or f"Resolved entity combining {len(members)} mentions."

        member_confidences = {}
        for h in history:
            if h.get("action") in ("merge_entities",):
                for mid in h.get("merged_ids", []):
                    if mid in members:
                        member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
            elif h.get("action") in ("rename_entity", "keep_entity", "kept_singleton"):
                mid = h.get("entity_id")
                if mid in members:
                    member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
            elif h.get("action") == "left_unchanged_low_sim":
                for mid in h.get("members", []):
                    if mid in members:
                        member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
        cluster_mean = mean_pairwise_sim(members, entities_by_id, emb_name, emb_desc, emb_ctx)
        for m in members:
            member_confidences.setdefault(m, float(cluster_mean))

        new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
        aliases = list({ (entities_by_id[m].get("entity_name") or "") for m in members if (entities_by_id[m].get("entity_name") or "").strip().lower() != canonical_name.strip().lower() })
        resolved = {
            "id_final": new_id,
            "label": canonical_name,
            "aliases": aliases,
            "description": description,
            "members": members,
            "member_confidence": { m: member_confidences.get(m, 0.0) for m in members },
            "flag": "resolved_entity"
        }
        resolved_records.append(resolved)
        for m in members:
            resolve_map[m] = new_id
        clusters_processed.append({
            "root": root,
            "members": members,
            "label": canonical_name,
            "mean_pairwise_sim": cluster_mean,
            "member_confidence": { m: member_confidences.get(m, 0.0) for m in members }
        })

    ENT_OUT = OUT_DIR / "entities_resolved.jsonl"
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for r in resolved_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    MAP_OUT = OUT_DIR / "resolve_map.json"
    with open(MAP_OUT, "w", encoding="utf-8") as fh:
        json.dump(resolve_map, fh, indent=2, ensure_ascii=False)

    HISTORY_OUT = OUT_DIR / "resolution_history.jsonl"
    with open(HISTORY_OUT, "w", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")

    CLUSTERS_OUT = OUT_DIR / "clusters_processed.jsonl"
    with open(CLUSTERS_OUT, "w", encoding="utf-8") as fh:
        for c in clusters_processed:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    if VERBOSE:
        print("[done] results written to:", OUT_DIR)
        print(" - entities_resolved:", ENT_OUT)
        print(" - resolve_map:", MAP_OUT)
        print(" - history:", HISTORY_OUT)
        print(" - clusters_processed:", CLUSTERS_OUT)
        print("Summary:")
        print(f"  original_mentions: {len(entities)}")
        print(f"  resolved_entities (canonical): {len(resolved_records)}")
        print(f"  total_llm_calls (approx): {total_llm_calls}")

if __name__ == "__main__":
    main()










#endregion#? Entity Resolution V1
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Rerun Entity Resolution V1


#!/usr/bin/env python3
"""
rerun_resolution_on_resolved.py

Collapse resolved mention-groups into representative nodes and re-run Entity Resolution
on representatives to catch cross-group merges that previous mention-level micro-batching
may have missed.

Outputs (OUT_DIR):
 - entities_resolved_rerun.jsonl
 - resolve_map_rerun.json
 - rerun_history.jsonl
 - debug files (per LLM call) in llm_debug_rerun/

This version includes robust numeric handling to avoid dtype/shape issues.
"""
import json, os, uuid, time, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

# ---------- CONFIG ----------
BASE_OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
RESOLVED_INPUT = BASE_OUT / "entities_resolved.jsonl"   # produced by your V1
ORIG_ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
OUT_DIR = BASE_OUT
RERUN_DEBUG = OUT_DIR / "llm_debug_rerun"
RERUN_DEBUG.mkdir(parents=True, exist_ok=True)

# Embedding settings (match your V1)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
USE_CUDA = True
BATCH_EMBED = 64

# thresholds (tunable)
T_BASE = 0.75
T_AUTO = 0.92       # auto-merge threshold for representatives (lower than mention-level typically)
MIN_SIM = 0.40
MAX_ROUNDS = 5      # collapse+rerun rounds
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_LLM_BATCH = 32   # representative count usually small; we can process full clusters

VERBOSE = True

# ---------- deps ----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError("Missing torch/transformers for embedding.") from e

try:
    import faiss
except Exception:
    try:
        import faiss_cpu as faiss
    except Exception:
        raise RuntimeError("Missing faiss or faiss_cpu.") from None

# try OpenAI client(s)
_OPENAI_CLIENT = None
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_CLIENT = ("openai_client", OpenAIClient)
except Exception:
    try:
        import openai
        _OPENAI_CLIENT = ("openai_pkg", openai)
    except Exception:
        _OPENAI_CLIENT = None

# ---------- util helpers ----------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            items.append(json.loads(ln))
    return items

def save_jsonl(path: Path, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def save_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def normalize(s: str) -> str:
    if not s: return ""
    return " ".join(s.strip().lower().split())

# ---------- embedding (same approach as V1) ----------
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
if VERBOSE:
    print(f"[init] loading embedding model {EMBED_MODEL} on {device}")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
model.eval()

import numpy as np
@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = BATCH_EMBED) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        token_emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        token_emb = token_emb * mask
        summed = token_emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / denom
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        vecs.append(mean_emb.cpu().numpy().astype("float32"))
    if vecs:
        return np.vstack(vecs)
    return np.zeros((0, model.config.hidden_size), dtype="float32")

# ---------- FAISS helpers ----------
def build_faiss_index(emb_map):
    ids = list(emb_map.keys())
    vecs = np.vstack([emb_map[i] for i in ids]).astype("float32")
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    return idx, ids, vecs

def neighbors_topn(index, ids_list, vecs, query_vec, topn=64):
    q = np.asarray(query_vec).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, topn)
    res = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(ids_list): continue
        res.append((ids_list[idx], float(dist)))
    return res

def safe_vector(x):
    """Return 1-D numpy float32 vector for x. If x is None or not convertible, return None."""
    if x is None:
        return None
    arr = np.asarray(x, dtype="float32")
    if arr.size == 0:
        return None
    return arr.ravel()

def safe_cosine(a, b) -> float:
    a_v = safe_vector(a)
    b_v = safe_vector(b)
    if a_v is None or b_v is None:
        return 0.0
    a_norm = np.linalg.norm(a_v)
    b_norm = np.linalg.norm(b_v)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a_v, b_v) / (a_norm * b_norm))

# ---------- simple composite sim for reps ----------
def composite_sim_rep(a: Dict, b: Dict, emb_map):
    # prefer embedding similarity if available, else token overlap heuristic
    va = emb_map.get(a["rep_id"])
    vb = emb_map.get(b["rep_id"])
    if va is not None and vb is not None:
        return safe_cosine(va, vb)
    an = normalize(a.get("entity_name","")); bn = normalize(b.get("entity_name",""))
    if not an or not bn: return 0.0
    sa = set(an.split()); sb = set(bn.split())
    inter = sa & sb
    uni = sa | sb
    return float(len(inter)/len(uni)) if uni else 0.0

# ---------- LLM helpers (robust JSON extraction) ----------
def extract_first_json_array(txt: str):
    if not txt: return []
    t = txt.strip()
    if t.startswith("```"): t = t.strip("`").strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list): return parsed
    except Exception:
        pass
    m = re.search(r'(\[\s*(?:[\s\S]*?)\s*\])', t)
    if m:
        candidate = m.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list): return parsed
        except Exception:
            try:
                parsed = json.loads(candidate.replace("'", '"'))
                if isinstance(parsed, list): return parsed
            except Exception:
                pass
    objs = []
    stack = []
    start = None
    for i,ch in enumerate(t):
        if ch == '{':
            if not stack: start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    piece = t[start:i+1]
                    try:
                        obj = json.loads(piece); objs.append(obj)
                    except Exception:
                        try:
                            obj = json.loads(piece.replace("'", '"')); objs.append(obj)
                        except Exception:
                            pass
                    start = None
    if objs: return objs
    try:
        parsed = json.loads(t.replace("'", '"'))
        if isinstance(parsed, list): return parsed
    except Exception:
        pass
    return []

def call_llm(prompt: str, model=OPENAI_MODEL, max_tokens=800, temperature=0.0):
    if not OPENAI_API_KEY or _OPENAI_CLIENT is None:
        if VERBOSE: print("[LLM] key/client not available - skipping")
        return None
    try:
        kind, client_pkg = _OPENAI_CLIENT
        if kind == "openai_client":
            client = client_pkg(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        else:
            client_pkg.api_key = OPENAI_API_KEY
            resp = client_pkg.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
    except Exception as e:
        print("[LLM] call error:", e)
        return None

# ---------- build prompt for representatives ----------
def build_prompt_for_rep_cluster(rep_ids: List[str], reps_map: Dict[str, Dict]):
    header = [
        "You are an assistant. Decide whether the following representative nodes refer to the same real-world entity.",
        "Return ONLY a JSON array with action objects. Each action must have a 'confidence' float 0.0-1.0.",
        "Allowed actions:",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - keep_entity: {action:'keep_entity', entity_id:'...', confidence:0.0-1.0}",
        ""
    ]
    items = []
    for rid in rep_ids:
        r = reps_map[rid]
        items.append({
            "rep_id": rid,
            "entity_name": r.get("entity_name"),
            "entity_description": (r.get("entity_description") or "")[:300],
            "members_count": len(r.get("members",[]))
        })
    prompt = "\n".join(header) + "\n\n" + json.dumps({"mentions": items}, ensure_ascii=False, indent=2)
    prompt += "\n\nReturn only a JSON array."
    return prompt

# ---------- main rerun logic ----------
def main():
    if not RESOLVED_INPUT.exists():
        raise FileNotFoundError(f"Resolved input not found: {RESOLVED_INPUT}")
    resolved = load_jsonl(RESOLVED_INPUT)
    if VERBOSE:
        print(f"[load] resolved records: {len(resolved)}")

    # Build representative nodes from resolved records
    reps = []
    for r in resolved:
        rep = {
            "rep_id": r.get("id_final"),
            "entity_name": r.get("label") or r.get("id_final"),
            "entity_description": r.get("description") or "",
            "members": r.get("members", [])
        }
        reps.append(rep)
    reps_map = { r["rep_id"]: r for r in reps }
    if VERBOSE:
        print(f"[reps] built {len(reps)} representatives")

    # Embed representatives (use name + description)
    texts = [ (r["entity_name"] or "") + " . " + (r["entity_description"] or "") for r in reps ]
    if texts:
        emb = embed_texts(texts)
    else:
        emb = np.zeros((0, model.config.hidden_size), dtype="float32")
    rep_emb_map = { reps[i]["rep_id"]: safe_vector(emb[i]) for i in range(len(reps)) }

    # iterative rounds: build FAISS on representatives, cluster, resolve
    round_no = 0
    overall_history = []
    changed_any = True

    while round_no < MAX_ROUNDS and changed_any:
        round_no += 1
        if VERBOSE: print(f"\n=== RERUN ROUND {round_no} ===")
        changed_any = False

        # build fast index (only for reps with vectors)
        if not rep_emb_map:
            if VERBOSE:
                print("[warn] no rep embeddings available; stopping")
            break
        fast_index, fast_ids, fast_vecs = build_faiss_index(rep_emb_map)

        # build similarity graph (undirected adjacency) using T_BASE
        adjacency = defaultdict(dict)
        for rid in list(reps_map.keys()):
            qvec = rep_emb_map.get(rid)
            if qvec is None:
                continue
            nbrs = neighbors_topn(fast_index, fast_ids, fast_vecs, qvec, topn=64)
            for nid, fast_score in nbrs:
                if nid == rid: continue
                v_nid = rep_emb_map.get(nid)
                if v_nid is None:
                    continue
                s = safe_cosine(qvec, v_nid)
                if s >= T_BASE:
                    adjacency[rid][nid] = s
                    adjacency[nid][rid] = s

        # connected components (representative clusters)
        visited = set()
        rep_clusters = []
        for rid in reps_map.keys():
            if rid in visited:
                continue
            stack = [rid]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited: continue
                visited.add(cur)
                comp.append(cur)
                for nbr in adjacency.get(cur, {}).keys():
                    if nbr not in visited:
                        stack.append(nbr)
            rep_clusters.append(sorted(comp))
        if VERBOSE:
            print(f"[graph] reps clusters: {len(rep_clusters)}")

        # process each rep-cluster
        for cl in rep_clusters:
            if len(cl) == 1:
                continue
            # compute mean pairwise sim
            sims = []
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    va = rep_emb_map.get(cl[i])
                    vb = rep_emb_map.get(cl[j])
                    if va is not None and vb is not None:
                        sims.append(safe_cosine(va, vb))
            mean_sim = float(sum(sims)/len(sims)) if sims else 0.0
            if mean_sim >= T_AUTO:
                # auto-merge: combine into first rep
                base = cl[0]
                involved = [rep_emb_map[x] for x in cl if x in rep_emb_map]
                if involved:
                    # compute average embedding safely
                    stacked = np.stack(involved, axis=0)
                    rep_emb_map[base] = np.mean(stacked, axis=0)
                # merge members and mark deletions
                for other in cl[1:]:
                    if other in reps_map:
                        reps_map[base]["members"].extend(reps_map[other].get("members", []))
                        reps_map[other]["_deleted"] = True
                        if other in rep_emb_map:
                            rep_emb_map.pop(other, None)
                changed_any = True
                msg = {"ts": time.time(), "action": "auto_merge_reps", "cluster": cl, "mean_sim": mean_sim}
                overall_history.append(msg)
                if VERBOSE:
                    print(f"[auto_merge_reps] merged cluster size {len(cl)} mean_sim={mean_sim:.3f}")
                continue

            # otherwise, if mean_sim >= MIN_SIM, invoke LLM to decide
            if mean_sim < MIN_SIM:
                if VERBOSE:
                    print(f"[skip] cluster mean_sim {mean_sim:.3f} < MIN_SIM")
                continue

            # build LLM prompt for the whole cluster (reps are few)
            prompt = build_prompt_for_rep_cluster(cl, reps_map)
            raw = call_llm(prompt)
            raw_retry = None
            actions = extract_first_json_array(raw) if raw else []
            if not actions and raw:
                retry_prompt = ("Your previous output was not valid JSON. Return ONLY a single JSON array of actions now. "
                                "Do not include any commentary. Here is your previous output for reference:\n\n" + (raw[:3000] if raw else ""))
                raw_retry = call_llm(retry_prompt)
                actions = extract_first_json_array(raw_retry) if raw_retry else []

            # save debug
            dbg = {"round": round_no, "cluster": cl, "prompt": prompt[:2000], "raw": raw or "", "raw_retry": raw_retry or ""}
            dbg_path = RERUN_DEBUG / f"rerun_debug_round{round_no}_{cl[0]}_{uuid.uuid4().hex[:6]}.json"
            with open(dbg_path, "w", encoding="utf-8") as fh:
                json.dump(dbg, fh, ensure_ascii=False, indent=2)

            if not actions:
                if VERBOSE:
                    print("[LLM] no valid actions returned; skipping cluster (kept)")
                overall_history.append({"ts": time.time(), "action": "llm_no_parse", "cluster": cl})
                continue

            # apply actions: merge_entities and keep_entity
            for act in actions:
                a = act.get("action")
                conf = float(act.get("confidence") or 0.0)
                if a == "merge_entities":
                    mids = act.get("merged_ids") or []
                    if len(mids) >= 2:
                        base = mids[0]
                        involved = [rep_emb_map[x] for x in mids if x in rep_emb_map]
                        if involved:
                            rep_emb_map[base] = np.mean(np.stack(involved, axis=0), axis=0)
                        for other in mids[1:]:
                            if other in reps_map:
                                reps_map[base]["members"].extend(reps_map[other].get("members", []))
                                reps_map[other]["_deleted"] = True
                                if other in rep_emb_map:
                                    rep_emb_map.pop(other, None)
                                changed_any = True
                        if VERBOSE:
                            print(f"[LLM] merged reps: {mids} conf={conf:.2f}")
                        overall_history.append({"ts": time.time(), "action":"llm_merge_reps", "merged": mids, "confidence": conf})
                elif a == "keep_entity":
                    overall_history.append({"ts": time.time(), "action":"llm_keep_rep", "entity": act.get("entity_id"), "confidence": float(act.get("confidence") or 0.0)})
                else:
                    overall_history.append({"ts": time.time(), "action":"llm_unknown", "payload": act})

        # cleanup deleted reps: actually remove from reps_map and rep_emb_map
        to_delete = [k for k,v in reps_map.items() if v.get("_deleted")]
        for d in to_delete:
            reps_map.pop(d, None)
            rep_emb_map.pop(d, None)
        if VERBOSE:
            print(f"[round {round_no}] finished. changed_any={changed_any}. reps remaining: {len(reps_map)}")

    # After rounds, produce new resolved_records and resolve_map that map original mentions to new resolved groups.
    new_resolved = []
    new_resolve_map = {}
    for rep_id, r in reps_map.items():
        new_id = f"ResEntR_{uuid.uuid4().hex[:8]}"
        members = list(dict.fromkeys(r.get("members", [])))  # unique, preserve order
        label = r.get("entity_name") or new_id
        desc = r.get("entity_description") or ""
        resolved_obj = {
            "id_final": new_id,
            "label": label,
            "description": desc,
            "members": members,
            "flag": "resolved_entity_rerun"
        }
        new_resolved.append(resolved_obj)
        for m in members:
            new_resolve_map[m] = new_id

    # save outputs
    ENT_OUT = OUT_DIR / "entities_resolved_rerun.jsonl"
    save_jsonl(ENT_OUT, new_resolved)
    MAP_OUT = OUT_DIR / "resolve_map_rerun.json"
    save_json(MAP_OUT, new_resolve_map)
    HISTORY_OUT = OUT_DIR / "rerun_history.jsonl"
    save_jsonl(HISTORY_OUT, overall_history)

    if VERBOSE:
        print("[done] rerun outputs:")
        print(" - entities_resolved_rerun:", ENT_OUT)
        print(" - resolve_map_rerun:", MAP_OUT)
        print(" - rerun_history:", HISTORY_OUT)
        print("summary: reps final:", len(reps_map), "rounds:", round_no)

if __name__ == "__main__":
    main()


#endregion#? Rerun Entity Resolution V1
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Run + Rerun



#!/usr/bin/env python3
"""
entity_resolution_merged.py

Merged Entity Resolution pipeline:
  Stage A: mention-level ER (iterative passes, auto merges + LLM micro-batches)
  Stage B: representative-level consolidation (collapse groups -> reps -> rerun)
  Output: final resolved entities, mapping, and history files.

Tune thresholds and models in the CONFIG section below.
"""

import json
import os
import re
import time
import uuid
import math
import itertools
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

# -------- CONFIG (tweak here) --------
INPUT_ENT_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model (HuggingFace)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
USE_CUDA = True
BATCH_EMBED = 64

# Similarity / retrieval thresholds
TOPN_FAST = 64
T_BASE = 0.75     # base edge threshold when building similarity graph
T_AUTO = 0.94     # auto-merge threshold at mention level
MIN_SIM = 0.60    # below this we won't attempt LLM/auto (left unchanged)

# Representative-level thresholds (after initial mention ER)
REP_T_AUTO = 0.92   # auto-merge threshold for representatives (a bit lower is OK)
REP_MIN_SIM = 0.35
REP_TOPN = 64

# LLM / passes
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # optional
MAX_LLM_BATCH = 6        # mention-level micro-batch
MAX_PASSES = 4            # mention-level passes
REP_MAX_ROUNDS = 5        # representative consolidation rounds

# Weights for composite similarity
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.25, "type": 0.05}

# Debugging / logging
VERBOSE = True
DEBUG_DIR = OUT_DIR / "llm_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
RERUN_DEBUG_DIR = OUT_DIR / "llm_debug_rerun"
RERUN_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
LLM_PARSE_RETRY = True

# ---------------- Dependent libs (transformers / torch / faiss) ----------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError("Missing transformers/torch. Install with `pip install torch transformers`") from e

try:
    import faiss
except Exception:
    try:
        import faiss_cpu as faiss
    except Exception:
        raise RuntimeError("Missing faiss (faiss-cpu) to continue.") from None

# optional sklearn (not required)
try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# OpenAI client attempt (optional)
_OPENAI_CLIENT = None
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_CLIENT = ("openai_client", OpenAIClient)
except Exception:
    try:
        import openai
        _OPENAI_CLIENT = ("openai_pkg", openai)
    except Exception:
        _OPENAI_CLIENT = None

# ---------------- Utilities ----------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    if VERBOSE:
        print(f"[load_entities] loaded {len(items)} entries from {path}")
    return items

def save_jsonl(path: Path, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

def save_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def normalize_surface(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

# ---------------- Numeric / vector helpers (safe) ----------------
import numpy as np

def safe_vector(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype="float32")
    if arr.size == 0:
        return None
    return arr.ravel()

def safe_cosine(a, b) -> float:
    a_v = safe_vector(a)
    b_v = safe_vector(b)
    if a_v is None or b_v is None:
        return 0.0
    a_norm = np.linalg.norm(a_v)
    b_norm = np.linalg.norm(b_v)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a_v, b_v) / (a_norm * b_norm))

# ---------------- Embeddings ----------------
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
if VERBOSE:
    print(f"[init] loading embedding model {EMBED_MODEL} on {device}")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
model.eval()

@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = BATCH_EMBED) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        token_emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        token_emb = token_emb * mask
        summed = token_emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / denom
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        vecs.append(mean_emb.cpu().numpy().astype("float32"))
    if vecs:
        return np.vstack(vecs)
    return np.zeros((0, model.config.hidden_size), dtype="float32")

def prepare_entity_embeddings(entities: List[Dict[str,Any]]):
    if VERBOSE:
        print("[embed] preparing name/desc/context texts for embedding")
    ids = [e["id"] for e in entities]
    name_texts = [ (e.get("entity_name") or "") for e in entities ]
    desc_texts = [ (e.get("entity_description") or "") for e in entities ]
    ctx_texts = [ ( (e.get("used_context_excerpt") or e.get("context_phrase") or "")[:512] ) for e in entities ]
    emb_name = embed_texts(name_texts)
    emb_desc = embed_texts(desc_texts)
    emb_ctx  = embed_texts(ctx_texts)
    emb_name_map = { ids[i]: emb_name[i] for i in range(len(ids)) }
    emb_desc_map = { ids[i]: emb_desc[i] for i in range(len(ids)) }
    emb_ctx_map  = { ids[i]: emb_ctx[i]  for i in range(len(ids)) }
    if VERBOSE:
        print(f"[embed] built embeddings: name:{len(emb_name_map)} desc:{len(emb_desc_map)} ctx:{len(emb_ctx_map)}")
    return emb_name_map, emb_desc_map, emb_ctx_map

# ---------------- Composite similarity ----------------
def composite_sim(e1: Dict, e2: Dict, emb_name, emb_desc, emb_ctx, weights=WEIGHTS):
    sims = []
    ws = []
    v1 = emb_name.get(e1["id"]); v2 = emb_name.get(e2["id"])
    if v1 is not None and v2 is not None:
        sims.append(safe_cosine(v1, v2)); ws.append(weights["name"])
    v1 = emb_desc.get(e1["id"]); v2 = emb_desc.get(e2["id"])
    if v1 is not None and v2 is not None and (np.any(v1) or np.any(v2)):
        sims.append(safe_cosine(v1, v2)); ws.append(weights["desc"])
    v1 = emb_ctx.get(e1["id"]); v2 = emb_ctx.get(e2["id"])
    if v1 is not None and v2 is not None and (np.any(v1) or np.any(v2)):
        sims.append(safe_cosine(v1, v2)); ws.append(weights["ctx"])
    t1 = (e1.get("entity_type_hint") or "").strip().lower()
    t2 = (e2.get("entity_type_hint") or "").strip().lower()
    if t1 and t2:
        sims.append(1.0 if t1 == t2 else 0.0); ws.append(weights["type"])
    if not sims:
        return 0.0
    total_w = sum(ws)
    if total_w <= 0:
        return sum(sims) / len(sims)
    weighted = sum(s * w for s, w in zip(sims, ws)) / total_w
    return float(weighted)

# ---------------- FAISS helpers ----------------
def build_faiss_index(emb_map: Dict[str, np.ndarray]):
    ids = list(emb_map.keys())
    if not ids:
        raise RuntimeError("Empty emb_map for FAISS index")
    vecs = np.vstack([emb_map[i] for i in ids]).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, ids, vecs

def neighbors_topn(index, ids_list, vecs, query_vec, topn=TOPN_FAST):
    q = np.asarray(query_vec).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, topn)
    res = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(ids_list): continue
        res.append((ids_list[idx], float(dist)))
    return res

# ---------------- Graph & clustering ----------------
def build_similarity_graph(entities, emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE):
    if VERBOSE:
        print("[graph] building candidate edges with base threshold", t_base)
    id_map = {e["id"]: e for e in entities}
    adjacency = defaultdict(dict)
    for e in entities:
        qid = e["id"]
        # pick best available vector
        # qvec = emb_name.get(qid) or emb_desc.get(qid) or emb_ctx.get(qid)
        # pick the best available vector (safe with numpy arrays)
        qvec = emb_name.get(qid)
        if qvec is None:
            qvec = emb_desc.get(qid)
        if qvec is None:
            qvec = emb_ctx.get(qid)
        # qvec will be None if no vector exists

        if qvec is None:
            continue
        nbrs = neighbors_topn(fast_index, fast_ids, fast_vecs, qvec, topn=TOPN_FAST)
        for nid, fast_score in nbrs:
            if nid == qid:
                continue
            s = composite_sim(e, id_map[nid], emb_name, emb_desc, emb_ctx)
            if s >= t_base:
                adjacency[qid][nid] = s
                adjacency[nid][qid] = s
    visited = set()
    clusters = []
    for eid in id_map.keys():
        if eid in visited:
            continue
        stack = [eid]; comp = []
        while stack:
            cur = stack.pop()
            if cur in visited: continue
            visited.add(cur)
            comp.append(cur)
            for nbr in adjacency.get(cur, {}).keys():
                if nbr not in visited:
                    stack.append(nbr)
        clusters.append(sorted(comp))
    if VERBOSE:
        print(f"[graph] built {len(clusters)} candidate clusters (including singletons)")
    return clusters, adjacency

# ---------------- Union-Find ----------------
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, a):
        if a not in self.parent:
            self.parent[a] = a
            return a
        if self.parent[a] == a:
            return a
        self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra
    def groups(self):
        d = {}
        for x in list(self.parent.keys()):
            r = self.find(x)
            d.setdefault(r, []).append(x)
        return list(d.values())

# ---------------- LLM prompt / call / parse ----------------
def build_cluster_prompt(cluster_entities: List[str], entities_by_id: Dict[str, Dict], examples: Optional[List[Dict]] = None):
    header = [
        "You are an assistant whose job is to decide whether entity mentions refer to the same real-world entity.",
        "Return ONLY a JSON array (no commentary). Each element is an action object.",
        "Action types:",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - rename_entity: {action:'rename_entity', entity_id:'...', new_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - keep_entity: {action:'keep_entity', entity_id:'...', confidence:0.0-1.0}",
        "Important: include 'confidence' (0.0-1.0) for each action.",
        ""
    ]
    examples_block = [
        {"action":"merge_entities","merged_ids":["En_A","En_B"],"canonical_name":"graphitization","new_description":"Formation of graphite in steel.","confidence":0.95},
        {"action":"rename_entity","entity_id":"En_C","new_name":"Heat_Fatigue","new_description":"Thermal fatigue cause.", "confidence":0.85},
        {"action":"keep_entity","entity_id":"En_D","confidence":0.40}
    ]
    header.append("EXAMPLES:")
    header.append(json.dumps(examples_block, ensure_ascii=False, indent=2))
    items = []
    for eid in cluster_entities:
        ent = entities_by_id[eid]
        items.append({
            "id": ent.get("id"),
            "surface": ent.get("entity_name"),
            "short_desc": (ent.get("entity_description") or "")[:200],
            "context_excerpt": (ent.get("used_context_excerpt") or ent.get("context_phrase") or "")[:300],
            "type_hint": ent.get("entity_type_hint")
        })
    prompt = "\n".join(header) + "\n\n" + json.dumps({"mentions": items}, ensure_ascii=False, indent=2)
    prompt += "\n\nReturn only a JSON array of actions."
    return prompt

def build_prompt_for_rep_cluster(rep_ids: List[str], reps_map: Dict[str, Dict]):
    header = [
        "You are an assistant. Decide whether the following representative nodes refer to the same real-world entity.",
        "Return ONLY a JSON array with action objects. Each action must have a 'confidence' float 0.0-1.0.",
        "Allowed actions:",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...', confidence:0.0-1.0}",
        " - keep_entity: {action:'keep_entity', entity_id:'...', confidence:0.0-1.0}",
        ""
    ]
    items = []
    for rid in rep_ids:
        r = reps_map[rid]
        items.append({
            "rep_id": rid,
            "entity_name": r.get("entity_name"),
            "entity_description": (r.get("entity_description") or "")[:300],
            "members_count": len(r.get("members",[]))
        })
    prompt = "\n".join(header) + "\n\n" + json.dumps({"mentions": items}, ensure_ascii=False, indent=2)
    prompt += "\n\nReturn only a JSON array."
    return prompt

def call_llm(prompt: str, model=OPENAI_MODEL, max_tokens=600, temperature=0.0):
    if not OPENAI_API_KEY or _OPENAI_CLIENT is None:
        if VERBOSE:
            print("[LLM] OpenAI key/client not available; skipping LLM call.")
        return None
    try:
        kind, client_pkg = _OPENAI_CLIENT
        if kind == "openai_client":
            client = client_pkg(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        else:
            client_pkg.api_key = OPENAI_API_KEY
            resp = client_pkg.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
    except Exception as e:
        print("[LLM] call error:", e)
        return None

def extract_first_json_array(txt: str):
    if not txt:
        return []
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    # 1) direct load
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 2) find first [...] block
    m = re.search(r'(\[\s*(?:[\s\S]*?)\s*\])', t)
    if m:
        candidate = m.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            try:
                parsed = json.loads(candidate.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    # 3) fallback: extract {...} objects
    objs = []
    stack = []
    start = None
    for i,ch in enumerate(t):
        if ch == '{':
            if not stack: start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    piece = t[start:i+1]
                    try:
                        obj = json.loads(piece)
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except Exception:
                        try:
                            obj = json.loads(piece.replace("'", '"'))
                            if isinstance(obj, dict):
                                objs.append(obj)
                        except Exception:
                            pass
                    start = None
    if objs:
        return objs
    # last attempt
    try:
        parsed = json.loads(t.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []

# ---------------- apply actions (mention-level) ----------------
def apply_actions(actions: List[Dict], entities_by_id: Dict[str, Dict], uf: UnionFind, history: List[Dict]):
    """
    Apply LLM actions to union-find and entities_by_id (rename suggestions).
    Returns True if any union change happened (i.e., changed structure).
    """
    changed = False
    for act in actions:
        if not isinstance(act, dict):
            history.append({"ts": time.time(), "action":"invalid_action_format", "payload":act})
            continue
        action = act.get("action")
        conf = float(act.get("confidence")) if act.get("confidence") is not None else None
        if action == "merge_entities":
            mids = act.get("merged_ids") or []
            if len(mids) >= 2:
                base = mids[0]
                for other in mids[1:]:
                    before_root = uf.find(other)
                    uf.union(base, other)
                    after_root = uf.find(other)
                    if before_root != after_root:
                        changed = True
            history.append({"ts": time.time(), "action":"merge_entities", "merged_ids":mids, "canonical_name":act.get("canonical_name"), "new_description":act.get("new_description"), "confidence":conf})
        elif action == "rename_entity":
            eid = act.get("entity_id")
            new_name = act.get("new_name")
            if eid in entities_by_id:
                entities_by_id[eid]["__rename_suggested"] = {"new_name": new_name, "new_description": act.get("new_description"), "confidence": conf}
            history.append({"ts": time.time(), "action":"rename_entity", "entity_id": eid, "new_name": new_name, "new_description": act.get("new_description"), "confidence": conf})
        elif action == "keep_entity":
            eid = act.get("entity_id")
            history.append({"ts": time.time(), "action":"keep_entity", "entity_id": eid, "confidence": conf})
        else:
            history.append({"ts": time.time(), "action":"unknown", "payload": act})
    return changed

# ---------------- cluster splitting ----------------
def split_cluster(cluster_ids: List[str], max_batch=MAX_LLM_BATCH):
    """
    Splits cluster into sequential windows of size max_batch.
    NOTE: we process each sub-window with the LLM, and apply union-find unions immediately,
    which allows merges across windows in later sub-windows (transitive).
    """
    if len(cluster_ids) <= max_batch:
        return [cluster_ids]
    out = []
    for i in range(0, len(cluster_ids), max_batch):
        out.append(cluster_ids[i:i+max_batch])
    return out

# ---------------- Representative consolidation helpers ----------------
def build_reps_from_groups(groups: Dict[str, List[str]], entities_by_id: Dict[str, Dict]):
    """
    groups: root -> list of mention ids
    Returns reps_map: rep_id -> {rep_id, entity_name, entity_description, members}
    """
    reps = {}
    for root, members in groups.items():
        # choose name from rename suggestions or most common normalized surface
        candidate_names = []
        candidate_confs = []
        for m in members:
            rs = entities_by_id[m].get("__rename_suggested")
            if rs and rs.get("new_name"):
                candidate_names.append(rs.get("new_name"))
                candidate_confs.append(float(rs.get("confidence") or 0.0))
        if candidate_names:
            if candidate_confs:
                chosen = candidate_names[int(np.argmax(candidate_confs))]
            else:
                chosen = Counter(candidate_names).most_common(1)[0][0]
        else:
            surfaces = [entities_by_id[m].get("entity_name") or "" for m in members]
            normalized = [normalize_surface(s) for s in surfaces if s]
            if normalized:
                chosen = Counter(normalized).most_common(1)[0][0]
            else:
                chosen = f"Rep_{root}"
        # description: pick any suggested new_description or first available
        descs = [ (entities_by_id[m].get("__rename_suggested") or {}).get("new_description") for m in members if (entities_by_id[m].get("__rename_suggested") or {}).get("new_description")]
        if descs:
            chosen_desc = descs[0]
        else:
            found = None
            for m in members:
                dd = entities_by_id[m].get("entity_description")
                if dd:
                    found = dd
                    break
            chosen_desc = found or ""
        rep_id = f"Rep_{uuid.uuid4().hex[:8]}"
        reps[rep_id] = {"rep_id": rep_id, "entity_name": chosen, "entity_description": chosen_desc, "members": list(members)}
    return reps

# ---------------- Main Pipeline ----------------
def main():
    # Load mentions
    mentions = load_jsonl(INPUT_ENT_PATH)
    entities_by_id = { e["id"]: e for e in mentions }

    # Build mention-level embeddings
    emb_name, emb_desc, emb_ctx = prepare_entity_embeddings(mentions)

    # Build a combined emb_map for FAISS (name preferred, fallback to desc/ctx)
    combined_emb_map = {}
    for eid, v in emb_name.items():
        combined_emb_map[eid] = v.copy()
    for eid, v in emb_desc.items():
        if eid not in combined_emb_map:
            combined_emb_map[eid] = v.copy()
        else:
            combined_emb_map[eid] = (combined_emb_map[eid] + v) / 2.0
    for eid, v in emb_ctx.items():
        if eid not in combined_emb_map:
            combined_emb_map[eid] = v.copy()

    if not combined_emb_map:
        raise RuntimeError("No embeddings available to build FAISS index; aborting")

    fast_index, fast_ids, fast_vecs = build_faiss_index(combined_emb_map)

    # union-find to apply merges
    uf = UnionFind()
    for eid in entities_by_id.keys():
        uf.find(eid)

    history = []
    total_llm_calls = 0
    pass_no = 0
    changed_in_pass = True

    # ---------- Stage A: mention-level iterative passes ----------
    while pass_no < MAX_PASSES and changed_in_pass:
        pass_no += 1
        if VERBOSE:
            print(f"\n=== PASS {pass_no} (mention-level) ===")
        changed_in_pass = False

        clusters, adjacency = build_similarity_graph(list(entities_by_id.values()), emb_name, emb_desc, emb_ctx, fast_index, fast_ids, fast_vecs, t_base=T_BASE)

        clusters_to_process = []
        for cl in clusters:
            mps = mean_pairwise_sim(cl, entities_by_id, emb_name, emb_desc, emb_ctx)
            if len(cl) == 1:
                eid = cl[0]
                history.append({"ts":time.time(),"action":"kept_singleton","entity_id":eid,"mean_sim":mps,"processed_by":"none","confidence":mps})
                continue
            if mps < MIN_SIM:
                history.append({"ts":time.time(),"action":"left_unchanged_low_sim","members":cl,"mean_sim":mps,"processed_by":"none","confidence":mps})
                continue
            clusters_to_process.append((cl, mps))

        if VERBOSE:
            print(f"[pass {pass_no}] clusters_to_process: {len(clusters_to_process)} (min_sim={MIN_SIM}, auto_th={T_AUTO})")

        for cl, mps in clusters_to_process:
            # skip if already unified to single root
            roots = set(uf.find(x) for x in cl)
            if len(roots) == 1:
                continue
            # auto-merge if very high mean sim
            if mps >= T_AUTO:
                for i in range(1, len(cl)):
                    uf.union(cl[0], cl[i])
                history.append({"ts":time.time(),"action":"auto_merge","members":cl,"mean_sim":mps,"processed_by":"auto","confidence":1.0})
                changed_in_pass = True
                if VERBOSE:
                    print(f"[auto_merge] merged cluster size {len(cl)} mean_sim={mps:.3f}")
                continue

            # LLM micro-batches (split cluster into windows)
            subparts = split_cluster(cl, max_batch=MAX_LLM_BATCH)
            for idx, sub in enumerate(subparts, start=1):
                # only active (not already merged into another root)
                sub_active = [s for s in sub if uf.find(s) == s]
                if not sub_active:
                    continue
                prompt = build_cluster_prompt(sub_active, entities_by_id, examples=None)
                total_llm_calls += 1
                raw = call_llm(prompt)
                raw_retry = None
                actions = extract_first_json_array(raw) if raw else []
                if not actions and raw and LLM_PARSE_RETRY:
                    retry_prompt = ("Your previous output was not valid JSON. Return ONLY a single JSON array (no commentary) with actions now. "
                                    "Previous output (for context):\n" + raw[:4000])
                    raw_retry = call_llm(retry_prompt)
                    actions = extract_first_json_array(raw_retry) if raw_retry else []
                # save debug file
                dbg_name = f"pass{pass_no}_cluster_{cl[0]}_sub{idx}_{uuid.uuid4().hex[:6]}.json"
                dbg_path = DEBUG_DIR / dbg_name
                with open(dbg_path, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"pass":pass_no,"cluster_root":cl[0],"sub_idx":idx,"prompt":prompt[:2000],"raw":raw or "","raw_retry":raw_retry or ""}, ensure_ascii=False, indent=2))
                if not actions:
                    # conservative fallback: mark keeps with low confidence
                    for s in sub_active:
                        history.append({"ts":time.time(),"action":"llm_parse_failed_fallback_keep","entity_id":s,"processed_by":"fallback","confidence":0.25})
                    if VERBOSE:
                        print(f"[LLM parse] failed for sub-batch {idx}/{len(subparts)}; fallback keeps")
                    continue
                # apply LLM actions immediately (unions applied now)
                applied = apply_actions(actions, entities_by_id, uf, history)
                if applied:
                    changed_in_pass = True
                if VERBOSE:
                    print(f"[LLM] processed sub-batch {idx}/{len(subparts)} for cluster(size={len(cl)}) -> actions: {len(actions)}")

        if VERBOSE:
            print(f"[pass {pass_no}] finished. changed_in_pass={changed_in_pass}")

    # After mention-level passes, assemble groups
    groups = {}
    for eid in entities_by_id.keys():
        root = uf.find(eid)
        groups.setdefault(root, []).append(eid)

    # Create resolved records (intermediate)
    resolved_records = []
    resolve_map = {}
    clusters_processed = []
    for root, members in groups.items():
        # canonical name selection (based on rename suggestions or most common normalized surface)
        candidate_names = []
        candidate_confs = []
        for m in members:
            rs = entities_by_id[m].get("__rename_suggested")
            if rs and rs.get("new_name"):
                candidate_names.append(rs.get("new_name"))
                candidate_confs.append(float(rs.get("confidence") or 0.0))
        if candidate_names:
            if candidate_confs:
                name = candidate_names[int(np.argmax(candidate_confs))]
            else:
                name = Counter(candidate_names).most_common(1)[0][0]
        else:
            surfaces = [entities_by_id[m].get("entity_name") or "" for m in members]
            normalized = [normalize_surface(s) for s in surfaces if s]
            name = Counter(normalized).most_common(1)[0][0] if normalized else f"Resolved_{root}"
        # description
        descs = [ (entities_by_id[m].get("__rename_suggested") or {}).get("new_description") for m in members if (entities_by_id[m].get("__rename_suggested") or {}).get("new_description")]
        if descs:
            description = descs[0]
        else:
            description = next((entities_by_id[m].get("entity_description") for m in members if entities_by_id[m].get("entity_description")), "") or f"Resolved entity combining {len(members)} mentions."
        # member confidences from history
        member_confidences = {}
        for h in history:
            if h.get("action") == "merge_entities":
                for mid in h.get("merged_ids", []):
                    if mid in members:
                        member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
            elif h.get("action") in ("rename_entity","keep_entity","kept_singleton"):
                mid = h.get("entity_id")
                if mid in members:
                    member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
            elif h.get("action") == "left_unchanged_low_sim":
                for mid in h.get("members", []):
                    if mid in members:
                        member_confidences[mid] = max(member_confidences.get(mid, 0.0), float(h.get("confidence") or 0.0))
        cluster_mean = mean_pairwise_sim(members, entities_by_id, emb_name, emb_desc, emb_ctx)
        for m in members:
            member_confidences.setdefault(m, float(cluster_mean))
        new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
        aliases = list({ (entities_by_id[m].get("entity_name") or "") for m in members if (entities_by_id[m].get("entity_name") or "").strip().lower() != name.strip().lower() })
        resolved_obj = {
            "id_final": new_id,
            "label": name,
            "aliases": aliases,
            "description": description,
            "members": members,
            "member_confidence": {m: member_confidences.get(m, 0.0) for m in members},
            "flag": "resolved_entity_stageA"
        }
        resolved_records.append(resolved_obj)
        for m in members:
            resolve_map[m] = new_id
        clusters_processed.append({"root": root, "members": members, "label": name, "mean_pairwise_sim": cluster_mean, "member_confidence": {m: member_confidences.get(m, 0.0) for m in members}})

    # Save intermediate outputs (optional)
    ENT_OUT_STAGEA = OUT_DIR / "entities_resolved_stageA.jsonl"
    save_jsonl(ENT_OUT_STAGEA, resolved_records)
    save_json(OUT_DIR / "resolve_map_stageA.json", resolve_map)
    save_jsonl(OUT_DIR / "resolution_history_stageA.jsonl", history)
    save_jsonl(OUT_DIR / "clusters_processed_stageA.jsonl", clusters_processed)
    if VERBOSE:
        print(f"[stageA done] stageA resolved groups: {len(resolved_records)} saved to {ENT_OUT_STAGEA}")

    # ---------- Stage B: representative-level consolidation (rerun) ----------
    # Build groups map root->members (we already have groups), make reps
    reps_map = build_reps_from_groups(groups, entities_by_id)

    # Build rep embeddings (name + description)
    rep_ids = list(reps_map.keys())
    rep_texts = [ (reps_map[r]["entity_name"] or "") + " . " + (reps_map[r]["entity_description"] or "") for r in rep_ids ]
    rep_emb = embed_texts(rep_texts) if rep_texts else np.zeros((0, model.config.hidden_size), dtype="float32")
    rep_emb_map = { rep_ids[i]: safe_vector(rep_emb[i]) for i in range(len(rep_ids)) } if len(rep_ids)>0 else {}

    round_no = 0
    changed_any = True
    overall_rerun_history = []

    while round_no < REP_MAX_ROUNDS and changed_any:
        round_no += 1
        if VERBOSE:
            print(f"\n=== REP RERUN ROUND {round_no} ===")
        changed_any = False
        if not rep_emb_map:
            if VERBOSE:
                print("[rep_rerun] no rep embeddings; stopping")
            break
        # build FAISS index for reps
        try:
            fast_idx, fast_ids, fast_vecs = build_faiss_index(rep_emb_map)
        except Exception as e:
            print("[rep_rerun] build faiss failed:", e)
            break
        # adjacency based on rep embeddings
        adjacency = defaultdict(dict)
        for rid in list(reps_map.keys()):
            qvec = rep_emb_map.get(rid)
            if qvec is None:
                continue
            nbrs = neighbors_topn(fast_idx, fast_ids, fast_vecs, qvec, topn=REP_TOPN)
            for nid, fast_score in nbrs:
                if nid == rid:
                    continue
                v_nid = rep_emb_map.get(nid)
                if v_nid is None:
                    continue
                s = safe_cosine(qvec, v_nid)
                if s >= T_BASE:
                    adjacency[rid][nid] = s
                    adjacency[nid][rid] = s
        # connected components at rep-level
        visited = set()
        rep_clusters = []
        for rid in list(reps_map.keys()):
            if rid in visited: continue
            stack = [rid]; comp = []
            while stack:
                cur = stack.pop()
                if cur in visited: continue
                visited.add(cur)
                comp.append(cur)
                for nbr in adjacency.get(cur, {}).keys():
                    if nbr not in visited:
                        stack.append(nbr)
            rep_clusters.append(sorted(comp))
        if VERBOSE:
            print(f"[graph] rep clusters: {len(rep_clusters)}")

        # process clusters (full cluster passed to LLM unless too large)
        for cl in rep_clusters:
            if len(cl) <= 1:
                continue
            # compute mean pairwise sim
            sims = []
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    sims.append(safe_cosine(rep_emb_map.get(cl[i]), rep_emb_map.get(cl[j])))
            mean_sim = float(sum(sims)/len(sims)) if sims else 0.0
            # auto-merge if very high
            if mean_sim >= REP_T_AUTO:
                base = cl[0]
                involved_vecs = [rep_emb_map[x] for x in cl if x in rep_emb_map and rep_emb_map[x] is not None]
                if involved_vecs:
                    rep_emb_map[base] = np.mean(np.stack(involved_vecs, axis=0), axis=0)
                for other in cl[1:]:
                    if other in reps_map:
                        reps_map[base]["members"].extend(reps_map[other].get("members", []))
                        reps_map[other]["_deleted"] = True
                        rep_emb_map.pop(other, None)
                changed_any = True
                overall_rerun_history.append({"ts":time.time(),"action":"auto_merge_reps","cluster":cl,"mean_sim":mean_sim})
                if VERBOSE:
                    print(f"[auto_merge_reps] merged cluster size {len(cl)} mean_sim={mean_sim:.3f}")
                continue
            # skip if too low
            if mean_sim < REP_MIN_SIM:
                if VERBOSE:
                    print(f"[rep skip] cluster mean_sim {mean_sim:.3f} < REP_MIN_SIM")
                continue
            # build prompt and call LLM for whole cluster (reps are small)
            prompt = build_prompt_for_rep_cluster(cl, reps_map)
            raw = call_llm(prompt)
            raw_retry = None
            actions = extract_first_json_array(raw) if raw else []
            if not actions and raw and LLM_PARSE_RETRY:
                retry_prompt = ("Your previous output was invalid. Return ONLY a JSON array of actions now. Previous output:\n" + raw[:3000])
                raw_retry = call_llm(retry_prompt)
                actions = extract_first_json_array(raw_retry) if raw_retry else []
            # save debug file
            dbg_path = RERUN_DEBUG_DIR / f"rerun_round{round_no}_{cl[0]}_{uuid.uuid4().hex[:6]}.json"
            with open(dbg_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"round":round_no,"cluster":cl,"prompt":prompt[:2000],"raw":raw or "","raw_retry":raw_retry or ""}, ensure_ascii=False, indent=2))
            if not actions:
                overall_rerun_history.append({"ts":time.time(),"action":"llm_no_parse_reps","cluster":cl})
                if VERBOSE:
                    print("[LLM] no valid actions for rep cluster; skipping")
                continue
            # apply actions
            for act in actions:
                a = act.get("action")
                conf = float(act.get("confidence") or 0.0)
                if a == "merge_entities":
                    mids = act.get("merged_ids") or []
                    if len(mids) >= 2:
                        base = mids[0]
                        involved_vecs = [rep_emb_map[x] for x in mids if x in rep_emb_map and rep_emb_map[x] is not None]
                        if involved_vecs:
                            rep_emb_map[base] = np.mean(np.stack(involved_vecs, axis=0), axis=0)
                        for other in mids[1:]:
                            if other in reps_map:
                                reps_map[base]["members"].extend(reps_map[other].get("members", []))
                                reps_map[other]["_deleted"] = True
                                rep_emb_map.pop(other, None)
                                changed_any = True
                        overall_rerun_history.append({"ts":time.time(),"action":"llm_merge_reps","merged":mids,"confidence":conf})
                        if VERBOSE:
                            print(f"[LLM] merged reps: {mids} conf={conf:.2f}")
                elif a == "keep_entity":
                    overall_rerun_history.append({"ts":time.time(),"action":"llm_keep_rep","entity":act.get("entity_id"),"confidence":float(act.get("confidence") or 0.0)})
                else:
                    overall_rerun_history.append({"ts":time.time(),"action":"llm_unknown_rep","payload":act})

        # cleanup deleted reps
        to_delete = [k for k,v in reps_map.items() if v.get("_deleted")]
        for d in to_delete:
            reps_map.pop(d, None)
            rep_emb_map.pop(d, None)
        if VERBOSE:
            print(f"[rep round {round_no}] finished. changed_any={changed_any}. reps remaining: {len(reps_map)}")

    # Build final resolved objects from reps_map
    final_resolved = []
    final_resolve_map = {}
    for rep_id, r in reps_map.items():
        new_id = f"ResEntFinal_{uuid.uuid4().hex[:8]}"
        members = list(dict.fromkeys(r.get("members", [])))  # unique preserve order
        label = r.get("entity_name") or new_id
        desc = r.get("entity_description") or ""
        obj = {"id_final": new_id, "label": label, "description": desc, "members": members, "flag":"resolved_entity_final"}
        final_resolved.append(obj)
        for m in members:
            final_resolve_map[m] = new_id

    # Save final outputs
    ENT_OUT = OUT_DIR / "entities_resolved_final.jsonl"
    save_jsonl(ENT_OUT, final_resolved)
    MAP_OUT = OUT_DIR / "resolve_map_final.json"
    save_json(MAP_OUT, final_resolve_map)
    HISTORY_OUT = OUT_DIR / "resolution_history_full.jsonl"
    # combine mention history and rerun history
    combined_history = history + overall_rerun_history
    save_jsonl(HISTORY_OUT, combined_history)

    if VERBOSE:
        print("[done] Final outputs:")
        print(" - entities_resolved_final:", ENT_OUT)
        print(" - resolve_map_final:", MAP_OUT)
        print(" - resolution_history_full:", HISTORY_OUT)
        print("Summary:")
        print(f"  original_mentions: {len(mentions)}")
        print(f"  resolved_entities (stageA): {len(resolved_records)}")
        print(f"  final_resolved_entities: {len(final_resolved)}")
        print(f"  total_llm_calls (approx): {total_llm_calls}")
    return

# ---------------- small helper used earlier ----------------
def mean_pairwise_sim(cluster_list: List[str], entities_by_id: Dict[str, Dict], emb_name, emb_desc, emb_ctx):
    if len(cluster_list) <= 1:
        return 1.0
    sims = []
    for a,b in itertools.combinations(cluster_list, 2):
        sims.append(composite_sim(entities_by_id[a], entities_by_id[b], emb_name, emb_desc, emb_ctx))
    return float(sum(sims)/len(sims)) if sims else 0.0

# ---------------- run main ----------------
if __name__ == "__main__":
    main()


#endregion#? Run + Rerun
#?#########################  End  ##########################


















#?######################### Start ##########################
#region:#?   Resolution Analysis Full V0


#!/usr/bin/env python3
"""
analyze_entity_resolution_full.py

Enhanced analysis / human-inspection outputs for entity resolution run.

Inputs (expected):
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/entities_resolved.jsonl
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/resolve_map.json
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/clusters_for_review.jsonl (optional)

Outputs (created under):
  /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/analysis/
    - resolved_entity_table.csv
    - merged_groups_full.json
    - merged_groups_full.jsonl
    - merged_groups_by_size.csv
    - singletons_unresolved.csv
    - clusters_flagged_for_review.jsonl (if present)
    - top_resolved_stats.json
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math

# --- paths: adjust if you moved files ---
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTest.jsonl")
RESOLVED_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/entities_resolved.jsonl")
RESOLVE_MAP = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/resolve_map.json")
CLUSTERS_REVIEW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/clusters_for_review.jsonl")

OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- helpers ---
def load_jsonl(path: Path):
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
            except Exception:
                # tolerant fallback: try single-quote replacement
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"Warning: failed to parse line in {path}: {ln[:100]}")
    return items

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

# --- load inputs ---
print("Loading files...")
entities_raw = load_jsonl(ENT_RAW)
print(f"  original mentions loaded: {len(entities_raw)}")

resolved_records = load_jsonl(RESOLVED_JSONL)
print(f"  resolved canonical records loaded: {len(resolved_records)}")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f"  resolve_map entries loaded: {len(resolve_map)}")

clusters_for_review = load_jsonl(CLUSTERS_REVIEW)
print(f"  clusters_for_review entries loaded: {len(clusters_for_review)}")

# quick lookups
entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_newid = { r.get("id_final"): r for r in resolved_records }

# ensure resolved records have expected fields
for r in resolved_by_newid.values():
    r.setdefault("label", r.get("label") or r.get("id_final"))
    r.setdefault("aliases", r.get("aliases") or [])
    r.setdefault("members", r.get("members") or [])
    r.setdefault("description", r.get("description") or "")

# --- 1) resolved_entity_table.csv (one row per original mention with resolved info) ---
csv_path = OUT_DIR / "resolved_entity_table.csv"
fields = [
    "original_id", "resolved_id", "resolved_label", "resolved_size", "resolved_aliases",
    "entity_name", "entity_description", "entity_type_hint", "chunk_id", "confidence_score"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for ent in entities_raw:
        eid = ent.get("id")
        rid = resolve_map.get(eid)
        rlabel = ""
        rsize = ""
        raliases = ""
        if rid and rid in resolved_by_newid:
            rec = resolved_by_newid[rid]
            rlabel = rec.get("label","")
            rsize = len(rec.get("members",[]))
            raliases = "|".join(rec.get("aliases",[]))
        row = {
            "original_id": eid,
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases,
            "entity_name": ent.get("entity_name",""),
            "entity_description": ent.get("entity_description",""),
            "entity_type_hint": ent.get("entity_type_hint",""),
            "chunk_id": ent.get("chunk_id") or ent.get("source_chunk") or "",
            "confidence_score": ent.get("confidence_score","")
        }
        writer.writerow(row)
print("Wrote:", csv_path)

# --- 2) merged_groups_full.json / .jsonl with member details (id,name,desc,type,chunk,confidence) ---
groups = defaultdict(list)
# build inverse map: resolved_id -> [orig ids]
for orig_id in entities_by_id:
    rid = resolve_map.get(orig_id)
    if rid:
        groups[rid].append(orig_id)
    else:
        # treat unresolved as its own pseudo-group (UNRESOLVED_<id>) if you want
        groups.setdefault(f"UNRESOLVED_{orig_id}", []).append(orig_id)

merged_groups = []
for rid, members in groups.items():
    # resolved fields if available
    resolved_rec = resolved_by_newid.get(rid)
    label = resolved_rec.get("label") if resolved_rec else None
    aliases = resolved_rec.get("aliases") if resolved_rec else []
    resolved_desc = resolved_rec.get("description") if resolved_rec else ""
    member_objs = []
    for mid in members:
        m = entities_by_id.get(mid, {})
        member_objs.append({
            "id": mid,
            "entity_name": m.get("entity_name",""),
            "entity_description": m.get("entity_description",""),
            "entity_type_hint": m.get("entity_type_hint",""),
            "chunk_id": m.get("chunk_id") or m.get("source_chunk") or "",
            "confidence_score": m.get("confidence_score"),
        })
    merged_groups.append({
        "resolved_id": rid,
        "label": label or f"UNRESOLVED_GROUP_{rid}",
        "aliases": aliases,
        "members": member_objs,
        "size": len(member_objs),
        "resolved_description": resolved_desc
    })

# sort by size desc
merged_groups_sorted = sorted(merged_groups, key=lambda x: x["size"], reverse=True)

out_json = OUT_DIR / "merged_groups_full.json"
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(merged_groups_sorted, fh, indent=2, ensure_ascii=False)
print("Wrote:", out_json)

out_jsonl = OUT_DIR / "merged_groups_full.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as fh:
    for g in merged_groups_sorted:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
print("Wrote:", out_jsonl)

# --- 3) merged_groups_by_size.csv (for quick sorting/filtering in Excel) ---
csv_groups = OUT_DIR / "merged_groups_by_size.csv"
with open(csv_groups, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "aliases", "sample_member_names"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        sample_names = "; ".join([m.get("entity_name","") for m in g["members"][:6]])
        writer.writerow({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "aliases": "|".join(g.get("aliases",[])),
            "sample_member_names": sample_names
        })
print("Wrote:", csv_groups)

# --- 4) singletons_unresolved.csv: mentions that remained single (size==1) but unresolved group key is UNRESOLVED_... ---
singletons_path = OUT_DIR / "singletons_unresolved.csv"
with open(singletons_path, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "confidence_score"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        if g["size"] == 1 and (str(g["resolved_id"]).startswith("UNRESOLVED_") or g["label"].startswith("UNRESOLVED_")):
            m = g["members"][0]
            writer.writerow({
                "original_id": m.get("id"),
                "entity_name": m.get("entity_name",""),
                "entity_description": m.get("entity_description",""),
                "entity_type_hint": m.get("entity_type_hint",""),
                "chunk_id": m.get("chunk_id",""),
                "confidence_score": m.get("confidence_score","")
            })
print("Wrote:", singletons_path)

# --- 5) clusters_flagged_for_review.jsonl copy/normalization ---
if clusters_for_review:
    out_review = OUT_DIR / "clusters_flagged_for_review.jsonl"
    with open(out_review, "w", encoding="utf-8") as fh:
        for item in clusters_for_review:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote review clusters copy:", out_review)
else:
    print("No clusters_for_review file present or it was empty; nothing to copy.")

# --- 6) top_resolved_stats.json ---
size_counts = Counter([g["size"] for g in merged_groups_sorted])
top_labels = [{"label":g["label"], "size":g["size"], "resolved_id": g["resolved_id"]} for g in merged_groups_sorted[:50]]
type_dist = Counter([ e.get("entity_type_hint") or "None" for e in entities_raw ])
confidence_vals = [ float(e.get("confidence_score")) for e in entities_raw if e.get("confidence_score") is not None ]
conf_summary = {}
if confidence_vals:
    conf_summary = {
        "count": len(confidence_vals),
        "mean": sum(confidence_vals)/len(confidence_vals),
        "min": min(confidence_vals),
        "max": max(confidence_vals)
    }

stats = {
    "n_original_mentions": len(entities_raw),
    "n_resolved_groups": len(merged_groups_sorted),
    "size_distribution": dict(size_counts),
    "top_resolved_samples": top_labels,
    "type_distribution_sample": type_dist.most_common(40),
    "confidence_summary": conf_summary
}
stats_out = OUT_DIR / "top_resolved_stats.json"
with open(stats_out, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
print("Wrote:", stats_out)

# --- final console summary ---
print("\n=== QUICK INSPECTION ===")
print("original mentions:", len(entities_raw))
print("resolved groups:", len(merged_groups_sorted))
print("top 10 resolved groups (label -- size -- resolved_id):")
for g in merged_groups_sorted[:10]:
    print(f"  {g['label'][:70]:70s}  size={g['size']:3d}  id={g['resolved_id']}")
print("\nFiles written to:", OUT_DIR)
print(" - resolved_entity_table.csv")
print(" - merged_groups_full.json(.jsonl)")
print(" - merged_groups_by_size.csv")
print(" - singletons_unresolved.csv")
print(" - top_resolved_stats.json")
if clusters_for_review:
    print(" - clusters_flagged_for_review.jsonl")
print("========================\n")


#endregion#? 
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Resolution Analysis Full V0 - merge source

#!/usr/bin/env python3
"""
annotate_merge_sources.py

Annotate merged groups with resolution source:
- auto
- llm
- mixed

Uses:
- merged_groups_full.json
- resolution_history.jsonl
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
ANALYSIS_DIR = BASE_DIR / "analysis"

MERGED_GROUPS = ANALYSIS_DIR / "merged_groups_full.json"
HISTORY = BASE_DIR / "resolution_history.jsonl"

OUT_JSON = ANALYSIS_DIR / "merged_groups_full_annotated.json"
OUT_JSONL = ANALYSIS_DIR / "merged_groups_full_annotated.jsonl"
OUT_CSV = ANALYSIS_DIR / "merged_groups_by_source.csv"

# ---------------- load ----------------
merged_groups = json.load(open(MERGED_GROUPS, "r", encoding="utf-8"))

history = []
with open(HISTORY, "r", encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if ln:
            history.append(json.loads(ln))

# ---------------- build entity → source map ----------------
entity_sources = defaultdict(list)

for h in history:
    action = h.get("action")

    if action == "auto_merge":
        for mid in h.get("member_ids", []):
            entity_sources[mid].append("auto")

    elif action == "merge_entities":
        for mid in h.get("merged_ids", []):
            entity_sources[mid].append("llm")

    elif action == "applied_merge_group":
        for mid in h.get("member_ids", []):
            entity_sources[mid].append("llm")

# ---------------- annotate groups ----------------
annotated = []
csv_rows = []

for g in merged_groups:
    members = g["members"]

    src_counter = Counter()
    for m in members:
        mid = m["id"]
        for src in entity_sources.get(mid, []):
            src_counter[src] += 1

    if not src_counter:
        resolution_source = "singleton"
    elif len(src_counter) == 1:
        resolution_source = next(iter(src_counter))
    else:
        resolution_source = "mixed"

    g2 = dict(g)
    g2["resolution_source"] = resolution_source
    g2["auto_merge_count"] = src_counter.get("auto", 0)
    g2["llm_merge_count"] = src_counter.get("llm", 0)

    annotated.append(g2)

    csv_rows.append({
        "resolved_id": g2["resolved_id"],
        "label": g2["label"],
        "size": g2["size"],
        "resolution_source": resolution_source,
        "auto_merge_count": g2["auto_merge_count"],
        "llm_merge_count": g2["llm_merge_count"],
        "sample_members": "; ".join(m["entity_name"] for m in members[:5])
    })

# ---------------- write outputs ----------------
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(annotated, fh, indent=2, ensure_ascii=False)

with open(OUT_JSONL, "w", encoding="utf-8") as fh:
    for g in annotated:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")

import csv
with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "resolved_id",
            "label",
            "size",
            "resolution_source",
            "auto_merge_count",
            "llm_merge_count",
            "sample_members"
        ]
    )
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)

print("Annotated merged groups written:")
print(" -", OUT_JSON)
print(" -", OUT_JSONL)
print(" -", OUT_CSV)



#endregion#? Resolution Analysis - merge source
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Resolution Analysis Full V0


#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
HISTORY = BASE / "resolution_history.jsonl"
CLUSTERS_REVIEW = BASE / "clusters_for_review.jsonl"
RES_MAP = BASE / "resolve_map.json"
ENT_RES = BASE / "entities_resolved.jsonl"

def load_jsonl(p):
    items=[]
    if not p.exists(): return items
    for ln in open(p,'r',encoding='utf-8'):
        ln=ln.strip()
        if ln: items.append(json.loads(ln))
    return items

hist = load_jsonl(HISTORY)
clusters_review = load_jsonl(CLUSTERS_REVIEW)
resolve_map = json.load(open(RES_MAP)) if RES_MAP.exists() else {}
resolved = load_jsonl(ENT_RES)

print("history entries:", len(hist))
# Count action types and sources
action_counter = Counter()
source_counter = Counter()
passes = set()
for i,h in enumerate(hist):
    action = h.get("action") or h.get("type") or "unknown"
    source = h.get("source") or h.get("by") or "unknown"
    action_counter[action]+=1
    source_counter[source]+=1
    # some history entries may include pass index -> collect if present
    if "pass" in h:
        passes.add(h["pass"])
print("action counts:", action_counter)
print("source counts:", source_counter)
print("distinct pass markers found in history:", sorted(list(passes)) if passes else "none")

# Which clusters were sent to LLM?
llm_clusters = [h for h in hist if h.get("action") in ("merge_entities","llm_merge","llm_decision","llm_action")]
print("LLM-related history events:", len(llm_clusters))
# Summarize merges by source
merge_by_src = defaultdict(int)
members_by_merge = []
for h in hist:
    if h.get("action") in ("merge_entities","llm_merge","auto_merge","applied_merge_group"):
        src = h.get("source","auto" if h.get("action")=="auto_merge" else "llm")
        merge_by_src[src]+=1
        members_by_merge.append({"src":src,"members": len(h.get("merged_ids", h.get("member_ids", [])))})
print("merge_by_source:", dict(merge_by_src))

# Print clusters_for_review if present
print("clusters_for_review exists:", CLUSTERS_REVIEW.exists())
if CLUSTERS_REVIEW.exists():
    print("clusters_for_review count:", len(clusters_review))
    for cr in clusters_review[:5]:
        print(" - sample flagged cluster:", cr.get("cluster_id") or cr.get("seed") or cr.get("members")[:5])

# Basic final counts
print("original mentions (estimated):", sum(len(v.get("members",[])) for v in resolved) if resolved else "unknown")
print("resolved canonical entities:", len(resolved) if resolved else "unknown")

# Show a few LLM history entries for inspection
print("\nSample LLM history rows (first 6):")
for row in hist[:6]:
    if row.get("source","").lower().startswith("llm") or row.get("action","").startswith("llm") or row.get("source")=="llm":
        print(json.dumps(row, indent=2))


#endregion#? Analysis
#?#########################  End  ##########################










#?######################### Start ##########################
#region:#?   Analysis V1





#!/usr/bin/env python3
"""
analyze_entity_resolution_combined.py

Combined analysis & annotation for the Entity Resolution pipeline.

Inputs (defaults, override via constants below):
  - ENT_RAW: original entity mentions (jsonl)
  - ENT_RESOLVED_JSONL: entities_resolved.jsonl (resolved canonical records)
  - RESOLVE_MAP: resolve_map.json (mapping from mention_id -> resolved_id)
  - HISTORY: resolution_history.jsonl (history of actions)
  - CLUSTERS_PROCESSED: clusters_processed.jsonl (optional audit info)
  - LLM_DEBUG_DIR: optional directory with LLM debug files

Outputs (written under OUT_DIR/analysis):
  - resolved_entity_table.csv
  - merged_groups_full.json
  - merged_groups_full.jsonl
  - merged_groups_by_size.csv
  - singletons_unresolved.csv
  - merged_groups_full_annotated.json / .jsonl / .csv (annotated by merge source)
  - top_resolved_stats.json
  - optional clusters_processed_copy.jsonl
  - prints a short console summary

Usage:
  python analyze_entity_resolution_combined.py
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import argparse

# ------------------ Config / paths (change if needed) ------------------
BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
# Default inputs (pipeline outputs)
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
ENT_RESOLVED_JSONL = BASE_DIR / "entities_resolved.jsonl"
RESOLVE_MAP = BASE_DIR / "resolve_map.json"
HISTORY = BASE_DIR / "resolution_history.jsonl"
CLUSTERS_PROCESSED = BASE_DIR / "clusters_processed.jsonl"   # optional audit info
CLUSTERS_FOR_REVIEW = BASE_DIR / "clusters_for_review.jsonl" # legacy/optional
LLM_DEBUG_DIR = BASE_DIR / "llm_debug"                       # optional

OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Helpers ------------------
def load_jsonl(path: Path):
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
            except Exception:
                # tolerant fallback: try single-quote replacement (best-effort)
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"[WARN] failed to parse line in {path}: {ln[:120]!r}")
    return items

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

# ------------------ Load inputs ------------------
print("Loading input files (defaults shown in script)...")
entities_raw = load_jsonl(ENT_RAW)
print(f" - original mentions: {len(entities_raw)}  (ENT_RAW: {ENT_RAW})")

resolved_records = load_jsonl(ENT_RESOLVED_JSONL)
print(f" - canonical resolved records: {len(resolved_records)}  (ENT_RESOLVED_JSONL: {ENT_RESOLVED_JSONL})")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f" - resolve_map entries: {len(resolve_map)}  (RESOLVE_MAP: {RESOLVE_MAP})")

history = load_jsonl(HISTORY)
print(f" - history entries: {len(history)}  (HISTORY: {HISTORY})")

clusters_processed = load_jsonl(CLUSTERS_PROCESSED)
if clusters_processed:
    print(f" - clusters_processed entries: {len(clusters_processed)}  (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")
else:
    print(f" - clusters_processed missing or empty (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")

clusters_for_review = load_jsonl(CLUSTERS_FOR_REVIEW)
if clusters_for_review:
    print(f" - clusters_for_review entries: {len(clusters_for_review)}  (CLUSTERS_FOR_REVIEW: {CLUSTERS_FOR_REVIEW})")

# Build lookups
entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_newid = { r.get("id_final"): r for r in resolved_records }

# Ensure resolved records have defaults
for r in resolved_by_newid.values():
    r.setdefault("label", r.get("label") or r.get("id_final"))
    r.setdefault("aliases", r.get("aliases") or [])
    r.setdefault("members", r.get("members") or [])
    r.setdefault("description", r.get("description") or "")

# ------------------ 1) resolved_entity_table.csv ------------------
csv_path = OUT_DIR / "resolved_entity_table.csv"
fields = [
    "original_id", "resolved_id", "resolved_label", "resolved_size", "resolved_aliases",
    "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence", "member_confidence"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for ent in entities_raw:
        eid = ent.get("id")
        rid = resolve_map.get(eid)
        rlabel = ""
        rsize = ""
        raliases = ""
        member_conf = None
        if rid and rid in resolved_by_newid:
            rec = resolved_by_newid[rid]
            rlabel = rec.get("label","")
            rsize = len(rec.get("members",[]))
            raliases = "|".join(rec.get("aliases",[]))
            member_conf = rec.get("member_confidence", {}).get(eid) if isinstance(rec.get("member_confidence", {}), dict) else None
        row = {
            "original_id": eid,
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases,
            "entity_name": ent.get("entity_name",""),
            "entity_description": ent.get("entity_description",""),
            "entity_type_hint": ent.get("entity_type_hint",""),
            "chunk_id": ent.get("chunk_id") or ent.get("source_chunk") or ent.get("chunk") or "",
            "mention_confidence": ent.get("confidence_score") if ent.get("confidence_score") is not None else "",
            "member_confidence": member_conf if member_conf is not None else ""
        }
        writer.writerow(row)
print("Wrote:", csv_path)

# ------------------ 2) merged_groups_full.json(.jsonl) ------------------
groups = defaultdict(list)
# Build groups: resolved_id -> orig ids; unresolved mentions become UNRESOLVED_<id>
for orig_id in entities_by_id:
    rid = resolve_map.get(orig_id)
    if rid:
        groups[rid].append(orig_id)
    else:
        groups[f"UNRESOLVED_{orig_id}"].append(orig_id)

merged_groups = []
for rid, members in groups.items():
    resolved_rec = resolved_by_newid.get(rid)
    label = resolved_rec.get("label") if resolved_rec else None
    aliases = resolved_rec.get("aliases") if resolved_rec else []
    resolved_desc = resolved_rec.get("description") if resolved_rec else ""
    member_objs = []
    for mid in members:
        m = entities_by_id.get(mid, {})
        member_objs.append({
            "id": mid,
            "entity_name": m.get("entity_name",""),
            "entity_description": m.get("entity_description",""),
            "entity_type_hint": m.get("entity_type_hint",""),
            "chunk_id": m.get("chunk_id") or m.get("source_chunk") or m.get("chunk") or "",
            "mention_confidence": m.get("confidence_score"),
            # member-level confidence recorded in resolved record if available
            "member_confidence": (resolved_by_newid.get(rid, {}).get("member_confidence", {}).get(mid) if rid in resolved_by_newid else None)
        })
    merged_groups.append({
        "resolved_id": rid,
        "label": label or f"UNRESOLVED_GROUP_{rid}",
        "aliases": aliases,
        "members": member_objs,
        "size": len(member_objs),
        "resolved_description": resolved_desc
    })

merged_groups_sorted = sorted(merged_groups, key=lambda x: x["size"], reverse=True)

out_json = OUT_DIR / "merged_groups_full.json"
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(merged_groups_sorted, fh, indent=2, ensure_ascii=False)
print("Wrote:", out_json)

out_jsonl = OUT_DIR / "merged_groups_full.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as fh:
    for g in merged_groups_sorted:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
print("Wrote:", out_jsonl)

# ------------------ 3) merged_groups_by_size.csv ------------------
csv_groups = OUT_DIR / "merged_groups_by_size.csv"
with open(csv_groups, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "aliases", "sample_member_names"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        sample_names = "; ".join([m.get("entity_name","") for m in g["members"][:6]])
        writer.writerow({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "aliases": "|".join(g.get("aliases",[])),
            "sample_member_names": sample_names
        })
print("Wrote:", csv_groups)

# ------------------ 4) singletons_unresolved.csv ------------------
singletons_path = OUT_DIR / "singletons_unresolved.csv"
with open(singletons_path, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        if g["size"] == 1 and (str(g["resolved_id"]).startswith("UNRESOLVED_") or g["label"].startswith("UNRESOLVED_")):
            m = g["members"][0]
            writer.writerow({
                "original_id": m.get("id"),
                "entity_name": m.get("entity_name",""),
                "entity_description": m.get("entity_description",""),
                "entity_type_hint": m.get("entity_type_hint",""),
                "chunk_id": m.get("chunk_id",""),
                "mention_confidence": m.get("mention_confidence","")
            })
print("Wrote:", singletons_path)

# ------------------ 5) clusters_for_review copy (if exists) ------------------
if clusters_for_review:
    out_review = OUT_DIR / "clusters_flagged_for_review.jsonl"
    with open(out_review, "w", encoding="utf-8") as fh:
        for item in clusters_for_review:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote review clusters copy:", out_review)

# ------------------ 6) annotate merge sources (auto/llm/mixed) ------------------
# Build entity -> sources map from history
entity_sources = defaultdict(list)
# We try to detect common action names used in pipeline: 'auto_merge', 'merge_entities', 'rename_entity', 'keep_entity', etc.
for h in history:
    act = h.get("action", "").lower()
    # try different field names used earlier
    if act == "auto_merge" or act == "auto_merge":
        # some old history used 'member_ids' vs 'merged_ids'
        mids = h.get("member_ids") or h.get("merged_ids") or h.get("members") or []
        for m in mids:
            entity_sources[m].append("auto")
    elif act in ("merge_entities", "merge_entities"):  # LLM merges
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act in ("applied_merge_group", "applied_merge"):
        mids = h.get("member_ids") or h.get("merged_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act == "rename_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    elif act == "kept_singleton":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("none")
    elif act == "left_unchanged_low_sim":
        for m in h.get("members", []) or []:
            entity_sources[m].append("none")
    elif act == "keep_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    # other history entries may also include merge data; we will inspect any 'merged_ids' or 'member_ids'
    else:
        # conservative: if merged_ids present mark as llm
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm" if act and "auto" not in act else "auto")

# Annotate groups by source
annotated = []
csv_rows = []
for g in merged_groups_sorted:
    members = g["members"]
    src_counter = Counter()
    for m in members:
        mid = m["id"]
        for src in entity_sources.get(mid, []):
            src_counter[src] += 1
    if not src_counter:
        resolution_source = "singleton"
    elif len(src_counter) == 1:
        resolution_source = next(iter(src_counter))
    else:
        # mixed sources possible (auto + llm)
        resolution_source = "mixed"
    g2 = dict(g)
    g2["resolution_source"] = resolution_source
    g2["auto_merge_count"] = src_counter.get("auto", 0)
    g2["llm_merge_count"] = src_counter.get("llm", 0)
    annotated.append(g2)
    csv_rows.append({
        "resolved_id": g2["resolved_id"],
        "label": g2["label"],
        "size": g2["size"],
        "resolution_source": resolution_source,
        "auto_merge_count": g2["auto_merge_count"],
        "llm_merge_count": g2["llm_merge_count"],
        "sample_members": "; ".join(m["entity_name"] for m in members[:5])
    })

OUT_ANNOT_JSON = OUT_DIR / "merged_groups_full_annotated.json"
OUT_ANNOT_JSONL = OUT_DIR / "merged_groups_full_annotated.jsonl"
OUT_ANNOT_CSV = OUT_DIR / "merged_groups_by_source.csv"

with open(OUT_ANNOT_JSON, "w", encoding="utf-8") as fh:
    json.dump(annotated, fh, indent=2, ensure_ascii=False)
with open(OUT_ANNOT_JSONL, "w", encoding="utf-8") as fh:
    for g in annotated:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(OUT_ANNOT_CSV, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "resolution_source", "auto_merge_count", "llm_merge_count", "sample_members"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)

print("Wrote annotated merged groups:", OUT_ANNOT_JSON, OUT_ANNOT_JSONL, OUT_ANNOT_CSV)

# ------------------ 7) top_resolved_stats.json ------------------
size_counts = Counter([g["size"] for g in merged_groups_sorted])
top_labels = [{"label":g["label"], "size":g["size"], "resolved_id": g["resolved_id"]} for g in merged_groups_sorted[:50]]
type_dist = Counter([ e.get("entity_type_hint") or "None" for e in entities_raw ])
confidence_vals = []
for e in entities_raw:
    cs = e.get("confidence_score")
    try:
        if cs is not None:
            confidence_vals.append(float(cs))
    except Exception:
        pass

conf_summary = {}
if confidence_vals:
    conf_summary = {
        "count": len(confidence_vals),
        "mean": sum(confidence_vals)/len(confidence_vals),
        "min": min(confidence_vals),
        "max": max(confidence_vals)
    }

stats = {
    "n_original_mentions": len(entities_raw),
    "n_resolved_groups": len(merged_groups_sorted),
    "size_distribution": dict(size_counts),
    "top_resolved_samples": top_labels,
    "type_distribution_sample": type_dist.most_common(40),
    "confidence_summary": conf_summary,
    "history_action_counts": Counter([h.get("action") for h in history])
}

stats_out = OUT_DIR / "top_resolved_stats.json"
with open(stats_out, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
print("Wrote:", stats_out)

# ------------------ Optional: save clusters_processed copy ------------------
if clusters_processed:
    cp_out = OUT_DIR / "clusters_processed_copy.jsonl"
    with open(cp_out, "w", encoding="utf-8") as fh:
        for c in clusters_processed:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Wrote clusters_processed copy:", cp_out)

# ------------------ Final console summary ------------------
print("\n=== QUICK INSPECTION ===")
print("original mentions:", len(entities_raw))
print("resolved groups:", len(merged_groups_sorted))
print("top 10 resolved groups (label -- size -- resolved_id):")
for g in merged_groups_sorted[:10]:
    print(f"  {g['label'][:70]:70s}  size={g['size']:3d}  id={g['resolved_id']}")
print("\nFiles written to:", OUT_DIR)
print(" - resolved_entity_table.csv")
print(" - merged_groups_full.json(.jsonl)")
print(" - merged_groups_by_size.csv")
print(" - singletons_unresolved.csv")
print(" - merged_groups_full_annotated.json(.jsonl/.csv)")
print(" - top_resolved_stats.json")
if clusters_for_review:
    print(" - clusters_flagged_for_review.jsonl")
if LLM_DEBUG_DIR.exists():
    print(" - LLM debug dir (exists):", LLM_DEBUG_DIR)
print("========================\n")





#endregion#? Analysis V1
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Analysis V2



#!/usr/bin/env python3
"""
analyze_entity_resolution_combined_with_nearmiss.py

Same as previous analysis script, but adds "near-miss" detection:
 - identifies mention-level pairs across different resolved groups whose
   textual similarity indicates they might deserve a merge (but weren't merged).
 - computes group-level similarity (representative labels) and suggests top group merges.

Outputs (in OUT_DIR):
 - near_miss_pairs.csv
 - near_miss_group_pairs.csv
 - merge_suggestions.jsonl
 + all previous analysis outputs (resolved_entity_table.csv, merged_groups_full.json, ...)

Notes:
 - Uses string-based similarity (difflib.SequenceMatcher + token Jaccard).
 - This is a pragmatic, lightweight heuristic; embedding-based similarity would be stronger.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import argparse
from difflib import SequenceMatcher
import itertools
import statistics

# ------------------ Config / paths (change if needed) ------------------
BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
ENT_RESOLVED_JSONL = BASE_DIR / "entities_resolved.jsonl"
RESOLVE_MAP = BASE_DIR / "resolve_map.json"
HISTORY = BASE_DIR / "resolution_history.jsonl"
CLUSTERS_PROCESSED = BASE_DIR / "clusters_processed.jsonl"
CLUSTERS_FOR_REVIEW = BASE_DIR / "clusters_for_review.jsonl"
LLM_DEBUG_DIR = BASE_DIR / "llm_debug"

OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Near-miss thresholds (string-similarity based)
NEAR_MISS_MIN = 0.70   # lower bound to consider "close"
NEAR_MISS_MAX = 0.94   # if > this we may have expected a merge (tunable)
TOP_K_SUGGESTIONS = 200  # how many top candidates to write to suggestions

# ------------------ Helpers ------------------
def load_jsonl(path: Path):
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
            except Exception:
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"[WARN] failed to parse line in {path}: {ln[:120]!r}")
    return items

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

def seq_ratio(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def token_jaccard(a: str, b: str) -> float:
    a_tokens = set([t for t in (a or "").lower().split() if t])
    b_tokens = set([t for t in (b or "").lower().split() if t])
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens.intersection(b_tokens)
    uni = a_tokens.union(b_tokens)
    return len(inter) / len(uni)

def composite_string_sim(name_a: str, name_b: str, desc_a: str, desc_b: str, weights=None):
    """
    Lightweight composite similarity using sequence ratio and token jaccard on names and descriptions.
    weights default tuned to prefer name similarity.
    """
    if weights is None:
        weights = {"name_seq":0.5, "name_jac":0.25, "desc_seq":0.15, "desc_jac":0.10}
    n_seq = seq_ratio(name_a, name_b)
    n_jac = token_jaccard(name_a, name_b)
    d_seq = seq_ratio(desc_a, desc_b)
    d_jac = token_jaccard(desc_a or "", desc_b or "")
    # normalize and weighted sum
    score = (n_seq * weights["name_seq"] + n_jac * weights["name_jac"] +
             d_seq * weights["desc_seq"] + d_jac * weights["desc_jac"])
    return float(score)

# ------------------ Load inputs ------------------
print("Loading input files (defaults shown in script)...")
entities_raw = load_jsonl(ENT_RAW)
print(f" - original mentions: {len(entities_raw)}  (ENT_RAW: {ENT_RAW})")

resolved_records = load_jsonl(ENT_RESOLVED_JSONL)
print(f" - canonical resolved records: {len(resolved_records)}  (ENT_RESOLVED_JSONL: {ENT_RESOLVED_JSONL})")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f" - resolve_map entries: {len(resolve_map)}  (RESOLVE_MAP: {RESOLVE_MAP})")

history = load_jsonl(HISTORY)
print(f" - history entries: {len(history)}  (HISTORY: {HISTORY})")

clusters_processed = load_jsonl(CLUSTERS_PROCESSED)
if clusters_processed:
    print(f" - clusters_processed entries: {len(clusters_processed)}  (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")
else:
    print(f" - clusters_processed missing or empty (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")

clusters_for_review = load_jsonl(CLUSTERS_FOR_REVIEW)
if clusters_for_review:
    print(f" - clusters_for_review entries: {len(clusters_for_review)}  (CLUSTERS_FOR_REVIEW: {CLUSTERS_FOR_REVIEW})")

entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_newid = { r.get("id_final"): r for r in resolved_records }

for r in resolved_by_newid.values():
    r.setdefault("label", r.get("label") or r.get("id_final"))
    r.setdefault("aliases", r.get("aliases") or [])
    r.setdefault("members", r.get("members") or [])
    r.setdefault("description", r.get("description") or "")

# ------------------ Existing analysis (unchanged) ------------------
csv_path = OUT_DIR / "resolved_entity_table.csv"
fields = [
    "original_id", "resolved_id", "resolved_label", "resolved_size", "resolved_aliases",
    "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence", "member_confidence"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for ent in entities_raw:
        eid = ent.get("id")
        rid = resolve_map.get(eid)
        rlabel = ""
        rsize = ""
        raliases = ""
        member_conf = None
        if rid and rid in resolved_by_newid:
            rec = resolved_by_newid[rid]
            rlabel = rec.get("label","")
            rsize = len(rec.get("members",[]))
            raliases = "|".join(rec.get("aliases",[]))
            member_conf = rec.get("member_confidence", {}).get(eid) if isinstance(rec.get("member_confidence", {}), dict) else None
        row = {
            "original_id": eid,
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases,
            "entity_name": ent.get("entity_name",""),
            "entity_description": ent.get("entity_description",""),
            "entity_type_hint": ent.get("entity_type_hint",""),
            "chunk_id": ent.get("chunk_id") or ent.get("source_chunk") or ent.get("chunk") or "",
            "mention_confidence": ent.get("confidence_score") if ent.get("confidence_score") is not None else "",
            "member_confidence": member_conf if member_conf is not None else ""
        }
        writer.writerow(row)
print("Wrote:", csv_path)

# Build groups
groups = defaultdict(list)
for orig_id in entities_by_id:
    rid = resolve_map.get(orig_id)
    if rid:
        groups[rid].append(orig_id)
    else:
        groups[f"UNRESOLVED_{orig_id}"].append(orig_id)

merged_groups = []
for rid, members in groups.items():
    resolved_rec = resolved_by_newid.get(rid)
    label = resolved_rec.get("label") if resolved_rec else None
    aliases = resolved_rec.get("aliases") if resolved_rec else []
    resolved_desc = resolved_rec.get("description") if resolved_rec else ""
    member_objs = []
    for mid in members:
        m = entities_by_id.get(mid, {})
        member_objs.append({
            "id": mid,
            "entity_name": m.get("entity_name",""),
            "entity_description": m.get("entity_description",""),
            "entity_type_hint": m.get("entity_type_hint",""),
            "chunk_id": m.get("chunk_id") or m.get("source_chunk") or m.get("chunk") or "",
            "mention_confidence": m.get("confidence_score"),
            "member_confidence": (resolved_by_newid.get(rid, {}).get("member_confidence", {}).get(mid) if rid in resolved_by_newid else None)
        })
    merged_groups.append({
        "resolved_id": rid,
        "label": label or f"UNRESOLVED_GROUP_{rid}",
        "aliases": aliases,
        "members": member_objs,
        "size": len(member_objs),
        "resolved_description": resolved_desc
    })

merged_groups_sorted = sorted(merged_groups, key=lambda x: x["size"], reverse=True)

out_json = OUT_DIR / "merged_groups_full.json"
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(merged_groups_sorted, fh, indent=2, ensure_ascii=False)
out_jsonl = OUT_DIR / "merged_groups_full.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as fh:
    for g in merged_groups_sorted:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
print("Wrote merged groups files.")

csv_groups = OUT_DIR / "merged_groups_by_size.csv"
with open(csv_groups, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "aliases", "sample_member_names"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        sample_names = "; ".join([m.get("entity_name","") for m in g["members"][:6]])
        writer.writerow({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "aliases": "|".join(g.get("aliases",[])),
            "sample_member_names": sample_names
        })
print("Wrote:", csv_groups)

singletons_path = OUT_DIR / "singletons_unresolved.csv"
with open(singletons_path, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        if g["size"] == 1 and (str(g["resolved_id"]).startswith("UNRESOLVED_") or g["label"].startswith("UNRESOLVED_")):
            m = g["members"][0]
            writer.writerow({
                "original_id": m.get("id"),
                "entity_name": m.get("entity_name",""),
                "entity_description": m.get("entity_description",""),
                "entity_type_hint": m.get("entity_type_hint",""),
                "chunk_id": m.get("chunk_id",""),
                "mention_confidence": m.get("mention_confidence","")
            })
print("Wrote:", singletons_path)

# ------------------ Annotate merge sources (unchanged) ------------------
entity_sources = defaultdict(list)
for h in history:
    act = (h.get("action") or "").lower()
    if act == "auto_merge":
        mids = h.get("member_ids") or h.get("merged_ids") or h.get("members") or []
        for m in mids:
            entity_sources[m].append("auto")
    elif act == "merge_entities":
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act in ("applied_merge_group", "applied_merge"):
        mids = h.get("member_ids") or h.get("merged_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act == "rename_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    elif act == "kept_singleton":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("none")
    elif act == "left_unchanged_low_sim":
        for m in h.get("members", []) or []:
            entity_sources[m].append("none")
    elif act == "keep_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    else:
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm" if act and "auto" not in act else "auto")

annotated = []
csv_rows = []
for g in merged_groups_sorted:
    members = g["members"]
    src_counter = Counter()
    for m in members:
        mid = m["id"]
        for src in entity_sources.get(mid, []):
            src_counter[src] += 1
    if not src_counter:
        resolution_source = "singleton"
    elif len(src_counter) == 1:
        resolution_source = next(iter(src_counter))
    else:
        resolution_source = "mixed"
    g2 = dict(g)
    g2["resolution_source"] = resolution_source
    g2["auto_merge_count"] = src_counter.get("auto", 0)
    g2["llm_merge_count"] = src_counter.get("llm", 0)
    annotated.append(g2)
    csv_rows.append({
        "resolved_id": g2["resolved_id"],
        "label": g2["label"],
        "size": g2["size"],
        "resolution_source": resolution_source,
        "auto_merge_count": g2["auto_merge_count"],
        "llm_merge_count": g2["llm_merge_count"],
        "sample_members": "; ".join(m["entity_name"] for m in members[:5])
    })

OUT_ANNOT_JSON = OUT_DIR / "merged_groups_full_annotated.json"
OUT_ANNOT_JSONL = OUT_DIR / "merged_groups_full_annotated.jsonl"
OUT_ANNOT_CSV = OUT_DIR / "merged_groups_by_source.csv"

with open(OUT_ANNOT_JSON, "w", encoding="utf-8") as fh:
    json.dump(annotated, fh, indent=2, ensure_ascii=False)
with open(OUT_ANNOT_JSONL, "w", encoding="utf-8") as fh:
    for g in annotated:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(OUT_ANNOT_CSV, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "resolution_source", "auto_merge_count", "llm_merge_count", "sample_members"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)
print("Wrote annotated merged groups.")

# ------------------ Stats (unchanged) ------------------
size_counts = Counter([g["size"] for g in merged_groups_sorted])
top_labels = [{"label":g["label"], "size":g["size"], "resolved_id": g["resolved_id"]} for g in merged_groups_sorted[:50]]
type_dist = Counter([ e.get("entity_type_hint") or "None" for e in entities_raw ])
confidence_vals = []
for e in entities_raw:
    cs = e.get("confidence_score")
    try:
        if cs is not None:
            confidence_vals.append(float(cs))
    except Exception:
        pass

conf_summary = {}
if confidence_vals:
    conf_summary = {
        "count": len(confidence_vals),
        "mean": sum(confidence_vals)/len(confidence_vals),
        "min": min(confidence_vals),
        "max": max(confidence_vals)
    }

stats = {
    "n_original_mentions": len(entities_raw),
    "n_resolved_groups": len(merged_groups_sorted),
    "size_distribution": dict(size_counts),
    "top_resolved_samples": top_labels,
    "type_distribution_sample": type_dist.most_common(40),
    "confidence_summary": conf_summary,
    "history_action_counts": Counter([h.get("action") for h in history])
}

stats_out = OUT_DIR / "top_resolved_stats.json"
with open(stats_out, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
print("Wrote:", stats_out)

# ------------------ NEW: Near-miss detection ------------------

# Build representative label/desc for each group:
# If resolved record exists use label/desc; otherwise pick first member's name/desc.
group_reprs = {}
for g in merged_groups_sorted:
    rid = g["resolved_id"]
    label = g.get("label") or ""
    desc = g.get("resolved_description") or ""
    # fallback to first member
    if (not label or label.startswith("UNRESOLVED_")) and g["members"]:
        label = label if label and not label.startswith("UNRESOLVED_") else (g["members"][0].get("entity_name") or "")
    if (not desc) and g["members"]:
        desc = desc or (g["members"][0].get("entity_description") or "")
    group_reprs[rid] = {"label": str(label), "desc": str(desc), "size": g["size"], "members": [m["id"] for m in g["members"]]}

# 1) Group-level similarity (representative labels)
group_pairs = []
group_ids = list(group_reprs.keys())
for a, b in itertools.combinations(group_ids, 2):
    la = group_reprs[a]["label"]
    lb = group_reprs[b]["label"]
    da = group_reprs[a]["desc"]
    db = group_reprs[b]["desc"]
    score = composite_string_sim(la, lb, da, db)
    if score >= NEAR_MISS_MIN:
        group_pairs.append((a, b, score))

# sort and write group-level near-misses
group_pairs_sorted = sorted(group_pairs, key=lambda x: x[2], reverse=True)
gp_csv = OUT_DIR / "near_miss_group_pairs.csv"
with open(gp_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a", "group_b", "label_a", "label_b", "size_a", "size_b", "score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"],
            "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"],
            "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote group-level near-miss csv:", gp_csv)

# 2) Mention-level cross-group near-miss pairs (top K)
# We'll compute for mentions across differing resolved groups, but to limit cost do:
# - only compute pairs where names length > 1 and groups are different
# - collect top candidates by score
pairs = []
mention_items = []
# prepare mention list with resolved group id
for ent in entities_raw:
    mid = ent.get("id")
    rid = resolve_map.get(mid) or f"UNRES_{mid}"
    mention_items.append({
        "id": mid,
        "rid": rid,
        "name": ent.get("entity_name","") or "",
        "desc": ent.get("entity_description","") or ""
    })

# For efficiency, group mentions by group and compare group pairs that were flagged above first.
# If no group pairs flagged, compare all groups but limit size.
candidate_group_pairs = [ (a,b) for (a,b,sc) in group_pairs_sorted ]
if not candidate_group_pairs:
    # fallback: compare top N groups by size (to limit O(N^2))
    top_groups = [g["resolved_id"] for g in merged_groups_sorted[:50]]
    candidate_group_pairs = list(itertools.combinations(top_groups, 2))

# build map group->members (mention dicts)
group_to_mentions = defaultdict(list)
for m in mention_items:
    group_to_mentions[m["rid"]].append(m)

# compute mention-level scores for candidate group pairs
for a,b in candidate_group_pairs:
    membs_a = group_to_mentions.get(a, [])
    membs_b = group_to_mentions.get(b, [])
    # skip trivial empty groups
    if not membs_a or not membs_b:
        continue
    # compare cross pairs (limit by size to avoid explosion)
    limit_a = moun_a = len(membs_a)
    limit_b = len(membs_b)
    # cap per-group comparisons
    cap = 40
    membs_a_sample = membs_a[:cap]
    membs_b_sample = membs_b[:cap]
    for ma in membs_a_sample:
        for mb in membs_b_sample:
            name_score = seq_ratio(ma["name"], mb["name"])
            comp_score = composite_string_sim(ma["name"], mb["name"], ma["desc"], mb["desc"])
            if comp_score >= NEAR_MISS_MIN:
                pairs.append({
                    "mention_a": ma["id"], "mention_b": mb["id"],
                    "group_a": a, "group_b": b,
                    "name_a": ma["name"], "name_b": mb["name"],
                    "desc_a": ma["desc"], "desc_b": mb["desc"],
                    "score": comp_score,
                    "name_ratio": name_score
                })

# Keep only those in (NEAR_MISS_MIN, NEAR_MISS_MAX] to highlight "should-check" zone.
pairs_filtered = [p for p in pairs if p["score"] >= NEAR_MISS_MIN]
pairs_sorted = sorted(pairs_filtered, key=lambda x: x["score"], reverse=True)

pairs_csv = OUT_DIR / "near_miss_pairs.csv"
with open(pairs_csv, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["mention_a","mention_b","group_a","group_b","name_a","name_b","score","name_ratio"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for p in pairs_sorted:
        writer.writerow({
            "mention_a": p["mention_a"],
            "mention_b": p["mention_b"],
            "group_a": p["group_a"],
            "group_b": p["group_b"],
            "name_a": p["name_a"],
            "name_b": p["name_b"],
            "score": f"{p['score']:.4f}",
            "name_ratio": f"{p['name_ratio']:.4f}"
        })
print("Wrote mention-level near-miss pairs:", pairs_csv)

# 3) Merge suggestions: aggregate group-level + mention-level evidence
# For each group-pair compute a combined score: group_score + max_mention_score/2
group_evidence = defaultdict(list)
for p in pairs_sorted:
    key = tuple(sorted([p["group_a"], p["group_b"]]))
    group_evidence[key].append(p["score"])

suggestions = []
for (a,b,sc) in group_pairs_sorted:
    key = tuple(sorted([a,b]))
    mention_scores = group_evidence.get(key, [])
    best_m = max(mention_scores) if mention_scores else 0.0
    combined = sc * 0.6 + best_m * 0.4
    suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": sc, "best_mention_score": best_m, "combined_score": combined})

# also include group pairs that only came from mention evidence (not in group_pairs_sorted)
for key, mention_scores in group_evidence.items():
    if not any((key[0]==x[0] and key[1]==x[1]) or (key[0]==x[1] and key[1]==x[0]) for x in [(g[0],g[1]) for g in group_pairs_sorted]):
        best_m = max(mention_scores)
        combined = best_m
        suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": None, "best_mention_score": best_m, "combined_score": combined})

# sort suggestions by combined_score desc and write top-K
suggestions_sorted = sorted(suggestions, key=lambda x: (x["combined_score"] or 0.0), reverse=True)[:TOP_K_SUGGESTIONS]
sugg_out = OUT_DIR / "merge_suggestions.jsonl"
with open(sugg_out, "w", encoding="utf-8") as fh:
    for s in suggestions_sorted:
        fh.write(json.dumps(s, ensure_ascii=False) + "\n")
print("Wrote merge suggestions:", sugg_out)

# Also write near_miss_group_pairs top view (already saved above). Save top-K there as well.
gp_top_csv = OUT_DIR / "near_miss_group_pairs_top.csv"
with open(gp_top_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a","group_b","label_a","label_b","size_a","size_b","score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted[:TOP_K_SUGGESTIONS]:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"], "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"], "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote top group-level near-miss csv:", gp_top_csv)

# ------------------ Optional: Save clusters_processed copy ------------------
if clusters_processed:
    cp_out = OUT_DIR / "clusters_processed_copy.jsonl"
    with open(cp_out, "w", encoding="utf-8") as fh:
        for c in clusters_processed:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Wrote clusters_processed copy:", cp_out)

# ------------------ Final console summary ------------------
print("\n=== QUICK INSPECTION ===")
print("original mentions:", len(entities_raw))
print("resolved groups:", len(merged_groups_sorted))
print("top 10 resolved groups (label -- size -- resolved_id):")
for g in merged_groups_sorted[:10]:
    print(f"  {g['label'][:70]:70s}  size={g['size']:3d}  id={g['resolved_id']}")
print("\nFiles written to:", OUT_DIR)
print(" - resolved_entity_table.csv")
print(" - merged_groups_full.json(.jsonl)")
print(" - merged_groups_by_size.csv")
print(" - singletons_unresolved.csv")
print(" - merged_groups_full_annotated.json(.jsonl/.csv)")
print(" - top_resolved_stats.json")
print(" - near_miss_pairs.csv")
print(" - near_miss_group_pairs.csv  (and _top.csv)")
print(" - merge_suggestions.jsonl")
if clusters_for_review:
    print(" - clusters_flagged_for_review.jsonl")
if LLM_DEBUG_DIR.exists():
    print(" - LLM debug dir (exists):", LLM_DEBUG_DIR)
print("========================\n")


#endregion#? Analysis V2
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Analysis V3 - for merged resolution 



#!/usr/bin/env python3
"""
analyze_entity_resolution_combined_with_nearmiss_and_wrongmerges.py

Combined analysis:
 - original analysis outputs (resolved_entity_table.csv, merged_groups_full.json, ...)
 - near-miss detection (mention- and group-level) -> near_miss_pairs.csv, near_miss_group_pairs.csv, merge_suggestions.jsonl
 - NEW: potential "wrong merges" detection (low intra-group cohesion) -> wrong_merges.jsonl

Tunable thresholds near top of file.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import argparse
from difflib import SequenceMatcher
import itertools
import statistics
import sys

# ------------------ Config / paths (change if needed) ------------------
BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
ENT_RESOLVED_JSONL = BASE_DIR / "entities_resolved.jsonl"
RESOLVE_MAP = BASE_DIR / "resolve_map.json"
HISTORY = BASE_DIR / "resolution_history.jsonl"
CLUSTERS_PROCESSED = BASE_DIR / "clusters_processed.jsonl"
CLUSTERS_FOR_REVIEW = BASE_DIR / "clusters_for_review.jsonl"
LLM_DEBUG_DIR = BASE_DIR / "llm_debug"

OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Near-miss thresholds (string-similarity based)
NEAR_MISS_MIN = 0.70   # lower bound to consider "close"
NEAR_MISS_MAX = 0.94   # if > this we may have expected a merge (tunable)
TOP_K_SUGGESTIONS = 200  # how many top candidates to write to suggestions

# Wrong-merge detection thresholds
INTRA_COHESION_MIN = 0.55   # mean pairwise score below this -> flag as potential wrong merge
INTRA_PAIR_MIN = 0.40       # individual low-scoring pairs below this are noteworthy
MAX_MEMBER_PAIRS_REPORTED = 40  # cap number of pair details saved per flagged group

# ------------------ Helpers ------------------
def load_jsonl(path: Path):
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
            except Exception:
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"[WARN] failed to parse line in {path}: {ln[:120]!r}", file=sys.stderr)
    return items

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

def seq_ratio(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def token_jaccard(a: str, b: str) -> float:
    a_tokens = set([t for t in (a or "").lower().split() if t])
    b_tokens = set([t for t in (b or "").lower().split() if t])
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens.intersection(b_tokens)
    uni = a_tokens.union(b_tokens)
    return len(inter) / len(uni)

def composite_string_sim(name_a: str, name_b: str, desc_a: str, desc_b: str, weights=None):
    """
    Lightweight composite similarity using sequence ratio and token jaccard on names and descriptions.
    weights default tuned to prefer name similarity.
    """
    if weights is None:
        weights = {"name_seq":0.5, "name_jac":0.25, "desc_seq":0.15, "desc_jac":0.10}
    n_seq = seq_ratio(name_a, name_b)
    n_jac = token_jaccard(name_a, name_b)
    d_seq = seq_ratio(desc_a, desc_b)
    d_jac = token_jaccard(desc_a or "", desc_b or "")
    score = (n_seq * weights["name_seq"] + n_jac * weights["name_jac"] +
             d_seq * weights["desc_seq"] + d_jac * weights["desc_jac"])
    return float(score)

# ------------------ Load inputs ------------------
print("Loading input files (defaults shown in script)...")
entities_raw = load_jsonl(ENT_RAW)
print(f" - original mentions: {len(entities_raw)}  (ENT_RAW: {ENT_RAW})")

resolved_records = load_jsonl(ENT_RESOLVED_JSONL)
print(f" - canonical resolved records: {len(resolved_records)}  (ENT_RESOLVED_JSONL: {ENT_RESOLVED_JSONL})")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f" - resolve_map entries: {len(resolve_map)}  (RESOLVE_MAP: {RESOLVE_MAP})")

history = load_jsonl(HISTORY)
print(f" - history entries: {len(history)}  (HISTORY: {HISTORY})")

clusters_processed = load_jsonl(CLUSTERS_PROCESSED)
if clusters_processed:
    print(f" - clusters_processed entries: {len(clusters_processed)}  (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")
else:
    print(f" - clusters_processed missing or empty (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")

clusters_for_review = load_jsonl(CLUSTERS_FOR_REVIEW)
if clusters_for_review:
    print(f" - clusters_for_review entries: {len(clusters_for_review)}  (CLUSTERS_FOR_REVIEW: {CLUSTERS_FOR_REVIEW})")

entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_newid = { r.get("id_final"): r for r in resolved_records }

for r in resolved_by_newid.values():
    r.setdefault("label", r.get("label") or r.get("id_final"))
    r.setdefault("aliases", r.get("aliases") or [])
    r.setdefault("members", r.get("members") or [])
    r.setdefault("description", r.get("description") or "")

# ------------------ Existing analysis (unchanged) ------------------
csv_path = OUT_DIR / "resolved_entity_table.csv"
fields = [
    "original_id", "resolved_id", "resolved_label", "resolved_size", "resolved_aliases",
    "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence", "member_confidence"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for ent in entities_raw:
        eid = ent.get("id")
        rid = resolve_map.get(eid)
        rlabel = ""
        rsize = ""
        raliases = ""
        member_conf = None
        if rid and rid in resolved_by_newid:
            rec = resolved_by_newid[rid]
            rlabel = rec.get("label","")
            rsize = len(rec.get("members",[]))
            raliases = "|".join(rec.get("aliases",[]))
            member_conf = rec.get("member_confidence", {}).get(eid) if isinstance(rec.get("member_confidence", {}), dict) else None
        row = {
            "original_id": eid,
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases,
            "entity_name": ent.get("entity_name",""),
            "entity_description": ent.get("entity_description",""),
            "entity_type_hint": ent.get("entity_type_hint",""),
            "chunk_id": ent.get("chunk_id") or ent.get("source_chunk") or ent.get("chunk") or "",
            "mention_confidence": ent.get("confidence_score") if ent.get("confidence_score") is not None else "",
            "member_confidence": member_conf if member_conf is not None else ""
        }
        writer.writerow(row)
print("Wrote:", csv_path)

# Build groups
groups = defaultdict(list)
for orig_id in entities_by_id:
    rid = resolve_map.get(orig_id)
    if rid:
        groups[rid].append(orig_id)
    else:
        groups[f"UNRESOLVED_{orig_id}"].append(orig_id)

merged_groups = []
for rid, members in groups.items():
    resolved_rec = resolved_by_newid.get(rid)
    label = resolved_rec.get("label") if resolved_rec else None
    aliases = resolved_rec.get("aliases") if resolved_rec else []
    resolved_desc = resolved_rec.get("description") if resolved_rec else ""
    member_objs = []
    for mid in members:
        m = entities_by_id.get(mid, {})
        member_objs.append({
            "id": mid,
            "entity_name": m.get("entity_name",""),
            "entity_description": m.get("entity_description",""),
            "entity_type_hint": m.get("entity_type_hint",""),
            "chunk_id": m.get("chunk_id") or m.get("source_chunk") or m.get("chunk") or "",
            "mention_confidence": m.get("confidence_score"),
            "member_confidence": (resolved_by_newid.get(rid, {}).get("member_confidence", {}).get(mid) if rid in resolved_by_newid else None)
        })
    merged_groups.append({
        "resolved_id": rid,
        "label": label or f"UNRESOLVED_GROUP_{rid}",
        "aliases": aliases,
        "members": member_objs,
        "size": len(member_objs),
        "resolved_description": resolved_desc
    })

merged_groups_sorted = sorted(merged_groups, key=lambda x: x["size"], reverse=True)

out_json = OUT_DIR / "merged_groups_full.json"
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(merged_groups_sorted, fh, indent=2, ensure_ascii=False)
out_jsonl = OUT_DIR / "merged_groups_full.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as fh:
    for g in merged_groups_sorted:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
print("Wrote merged groups files.")

csv_groups = OUT_DIR / "merged_groups_by_size.csv"
with open(csv_groups, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "aliases", "sample_member_names"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        sample_names = "; ".join([m.get("entity_name","") for m in g["members"][:6]])
        writer.writerow({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "aliases": "|".join(g.get("aliases",[])),
            "sample_member_names": sample_names
        })
print("Wrote:", csv_groups)

singletons_path = OUT_DIR / "singletons_unresolved.csv"
with open(singletons_path, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        if g["size"] == 1 and (str(g["resolved_id"]).startswith("UNRESOLVED_") or g["label"].startswith("UNRESOLVED_")):
            m = g["members"][0]
            writer.writerow({
                "original_id": m.get("id"),
                "entity_name": m.get("entity_name",""),
                "entity_description": m.get("entity_description",""),
                "entity_type_hint": m.get("entity_type_hint",""),
                "chunk_id": m.get("chunk_id",""),
                "mention_confidence": m.get("mention_confidence","")
            })
print("Wrote:", singletons_path)

# ------------------ Annotate merge sources (unchanged) ------------------
entity_sources = defaultdict(list)
for h in history:
    act = (h.get("action") or "").lower()
    if act == "auto_merge":
        mids = h.get("member_ids") or h.get("merged_ids") or h.get("members") or []
        for m in mids:
            entity_sources[m].append("auto")
    elif act == "merge_entities":
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act in ("applied_merge_group", "applied_merge"):
        mids = h.get("member_ids") or h.get("merged_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act == "rename_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    elif act == "kept_singleton":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("none")
    elif act == "left_unchanged_low_sim":
        for m in h.get("members", []) or []:
            entity_sources[m].append("none")
    elif act == "keep_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    else:
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm" if act and "auto" not in act else "auto")

annotated = []
csv_rows = []
for g in merged_groups_sorted:
    members = g["members"]
    src_counter = Counter()
    for m in members:
        mid = m["id"]
        for src in entity_sources.get(mid, []):
            src_counter[src] += 1
    if not src_counter:
        resolution_source = "singleton"
    elif len(src_counter) == 1:
        resolution_source = next(iter(src_counter))
    else:
        resolution_source = "mixed"
    g2 = dict(g)
    g2["resolution_source"] = resolution_source
    g2["auto_merge_count"] = src_counter.get("auto", 0)
    g2["llm_merge_count"] = src_counter.get("llm", 0)
    annotated.append(g2)
    csv_rows.append({
        "resolved_id": g2["resolved_id"],
        "label": g2["label"],
        "size": g2["size"],
        "resolution_source": resolution_source,
        "auto_merge_count": g2["auto_merge_count"],
        "llm_merge_count": g2["llm_merge_count"],
        "sample_members": "; ".join(m["entity_name"] for m in members[:5])
    })

OUT_ANNOT_JSON = OUT_DIR / "merged_groups_full_annotated.json"
OUT_ANNOT_JSONL = OUT_DIR / "merged_groups_full_annotated.jsonl"
OUT_ANNOT_CSV = OUT_DIR / "merged_groups_by_source.csv"

with open(OUT_ANNOT_JSON, "w", encoding="utf-8") as fh:
    json.dump(annotated, fh, indent=2, ensure_ascii=False)
with open(OUT_ANNOT_JSONL, "w", encoding="utf-8") as fh:
    for g in annotated:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(OUT_ANNOT_CSV, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "resolution_source", "auto_merge_count", "llm_merge_count", "sample_members"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)
print("Wrote annotated merged groups.")

# ------------------ Stats (unchanged) ------------------
size_counts = Counter([g["size"] for g in merged_groups_sorted])
top_labels = [{"label":g["label"], "size":g["size"], "resolved_id": g["resolved_id"]} for g in merged_groups_sorted[:50]]
type_dist = Counter([ e.get("entity_type_hint") or "None" for e in entities_raw ])
confidence_vals = []
for e in entities_raw:
    cs = e.get("confidence_score")
    try:
        if cs is not None:
            confidence_vals.append(float(cs))
    except Exception:
        pass

conf_summary = {}
if confidence_vals:
    conf_summary = {
        "count": len(confidence_vals),
        "mean": sum(confidence_vals)/len(confidence_vals),
        "min": min(confidence_vals),
        "max": max(confidence_vals)
    }

stats = {
    "n_original_mentions": len(entities_raw),
    "n_resolved_groups": len(merged_groups_sorted),
    "size_distribution": dict(size_counts),
    "top_resolved_samples": top_labels,
    "type_distribution_sample": type_dist.most_common(40),
    "confidence_summary": conf_summary,
    "history_action_counts": Counter([h.get("action") for h in history])
}

stats_out = OUT_DIR / "top_resolved_stats.json"
with open(stats_out, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
print("Wrote:", stats_out)

# ------------------ NEW: Near-miss detection ------------------

# Build representative label/desc for each group:
group_reprs = {}
for g in merged_groups_sorted:
    rid = g["resolved_id"]
    label = g.get("label") or ""
    desc = g.get("resolved_description") or ""
    if (not label or label.startswith("UNRESOLVED_")) and g["members"]:
        label = (g["members"][0].get("entity_name") or "")
    if (not desc) and g["members"]:
        desc = (g["members"][0].get("entity_description") or "")
    group_reprs[rid] = {"label": str(label), "desc": str(desc), "size": g["size"], "members": [m["id"] for m in g["members"]]}

# 1) Group-level similarity (representative labels)
group_pairs = []
group_ids = list(group_reprs.keys())
for a, b in itertools.combinations(group_ids, 2):
    la = group_reprs[a]["label"]
    lb = group_reprs[b]["label"]
    da = group_reprs[a]["desc"]
    db = group_reprs[b]["desc"]
    score = composite_string_sim(la, lb, da, db)
    if score >= NEAR_MISS_MIN:
        group_pairs.append((a, b, score))

group_pairs_sorted = sorted(group_pairs, key=lambda x: x[2], reverse=True)
gp_csv = OUT_DIR / "near_miss_group_pairs.csv"
with open(gp_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a", "group_b", "label_a", "label_b", "size_a", "size_b", "score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"],
            "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"],
            "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote group-level near-miss csv:", gp_csv)

# 2) Mention-level cross-group near-miss pairs (top K)
pairs = []
mention_items = []
for ent in entities_raw:
    mid = ent.get("id")
    rid = resolve_map.get(mid) or f"UNRES_{mid}"
    mention_items.append({
        "id": mid,
        "rid": rid,
        "name": ent.get("entity_name","") or "",
        "desc": ent.get("entity_description","") or ""
    })

# Candidate group pairs: those flagged above first, fallback to top groups
candidate_group_pairs = [ (a,b) for (a,b,sc) in group_pairs_sorted ]
if not candidate_group_pairs:
    top_groups = [g["resolved_id"] for g in merged_groups_sorted[:50]]
    candidate_group_pairs = list(itertools.combinations(top_groups, 2))

group_to_mentions = defaultdict(list)
for m in mention_items:
    group_to_mentions[m["rid"]].append(m)

for a,b in candidate_group_pairs:
    membs_a = group_to_mentions.get(a, [])
    membs_b = group_to_mentions.get(b, [])
    if not membs_a or not membs_b:
        continue
    cap = 40
    membs_a_sample = membs_a[:cap]
    membs_b_sample = membs_b[:cap]
    for ma in membs_a_sample:
        for mb in membs_b_sample:
            name_score = seq_ratio(ma["name"], mb["name"])
            comp_score = composite_string_sim(ma["name"], mb["name"], ma["desc"], mb["desc"])
            if comp_score >= NEAR_MISS_MIN:
                pairs.append({
                    "mention_a": ma["id"], "mention_b": mb["id"],
                    "group_a": a, "group_b": b,
                    "name_a": ma["name"], "name_b": mb["name"],
                    "desc_a": ma["desc"], "desc_b": mb["desc"],
                    "score": comp_score,
                    "name_ratio": name_score
                })

pairs_filtered = [p for p in pairs if p["score"] >= NEAR_MISS_MIN]
pairs_sorted = sorted(pairs_filtered, key=lambda x: x["score"], reverse=True)

pairs_csv = OUT_DIR / "near_miss_pairs.csv"
with open(pairs_csv, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["mention_a","mention_b","group_a","group_b","name_a","name_b","score","name_ratio"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for p in pairs_sorted:
        writer.writerow({
            "mention_a": p["mention_a"],
            "mention_b": p["mention_b"],
            "group_a": p["group_a"],
            "group_b": p["group_b"],
            "name_a": p["name_a"],
            "name_b": p["name_b"],
            "score": f"{p['score']:.4f}",
            "name_ratio": f"{p['name_ratio']:.4f}"
        })
print("Wrote mention-level near-miss pairs:", pairs_csv)

# 3) Merge suggestions: aggregate group-level + mention-level evidence
group_evidence = defaultdict(list)
for p in pairs_sorted:
    key = tuple(sorted([p["group_a"], p["group_b"]]))
    group_evidence[key].append(p["score"])

suggestions = []
for (a,b,sc) in group_pairs_sorted:
    key = tuple(sorted([a,b]))
    mention_scores = group_evidence.get(key, [])
    best_m = max(mention_scores) if mention_scores else 0.0
    combined = sc * 0.6 + best_m * 0.4
    suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": sc, "best_mention_score": best_m, "combined_score": combined})

for key, mention_scores in group_evidence.items():
    found = any((key[0]==x[0] and key[1]==x[1]) for x in [(g[0],g[1]) for g in group_pairs_sorted])
    if not found:
        best_m = max(mention_scores)
        combined = best_m
        suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": None, "best_mention_score": best_m, "combined_score": combined})

suggestions_sorted = sorted(suggestions, key=lambda x: (x["combined_score"] or 0.0), reverse=True)[:TOP_K_SUGGESTIONS]
sugg_out = OUT_DIR / "merge_suggestions.jsonl"
with open(sugg_out, "w", encoding="utf-8") as fh:
    for s in suggestions_sorted:
        fh.write(json.dumps(s, ensure_ascii=False) + "\n")
print("Wrote merge suggestions:", sugg_out)

gp_top_csv = OUT_DIR / "near_miss_group_pairs_top.csv"
with open(gp_top_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a","group_b","label_a","label_b","size_a","size_b","score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted[:TOP_K_SUGGESTIONS]:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"], "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"], "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote top group-level near-miss csv:", gp_top_csv)

# ------------------ NEW: Potential wrong merges detection ------------------
# For each resolved group (size>1) compute intra-group pairwise composite_string_sim on member names/descs.
# If mean pairwise score < INTRA_COHESION_MIN, flag the group as a potential wrong merge.
wrong_merges = []
for g in merged_groups_sorted:
    members = g["members"]
    if len(members) <= 1:
        continue
    # compute all pairwise scores, but cap reporting
    pair_scores = []
    for a, b in itertools.combinations(members, 2):
        name_a = a.get("entity_name") or ""
        name_b = b.get("entity_name") or ""
        desc_a = a.get("entity_description") or ""
        desc_b = b.get("entity_description") or ""
        sc = composite_string_sim(name_a, name_b, desc_a, desc_b)
        pair_scores.append({
            "a_id": a.get("id"),
            "b_id": b.get("id"),
            "a_name": name_a,
            "b_name": name_b,
            "score": sc
        })
    if not pair_scores:
        continue
    mean_pair = sum(p["score"] for p in pair_scores) / len(pair_scores)
    # count how many extremely low pairs exist
    low_pairs = [p for p in pair_scores if p["score"] <= INTRA_PAIR_MIN]
    if mean_pair < INTRA_COHESION_MIN or len(low_pairs) > 0:
        # sort pair_scores by ascending (lowest score first) to show problematic pairs
        pair_scores_sorted = sorted(pair_scores, key=lambda x: x["score"])
        # trim details to avoid huge files
        detail_pairs = pair_scores_sorted[:MAX_MEMBER_PAIRS_REPORTED]
        wrong_merges.append({
            "resolved_id": g["resolved_id"],
            "label": g.get("label"),
            "size": g.get("size"),
            "mean_pairwise_score": mean_pair,
            "n_pairs_total": len(pair_scores),
            "n_low_pairs": len(low_pairs),
            "low_pair_examples": detail_pairs
        })

# Save wrong merges JSONL and JSON summary
wrong_jsonl = OUT_DIR / "wrong_merges.jsonl"
with open(wrong_jsonl, "w", encoding="utf-8") as fh:
    for w in wrong_merges:
        fh.write(json.dumps(w, ensure_ascii=False) + "\n")
wrong_json = OUT_DIR / "wrong_merges_summary.json"
with open(wrong_json, "w", encoding="utf-8") as fh:
    json.dump({"n_flagged_groups": len(wrong_merges), "flags": wrong_merges}, fh, indent=2, ensure_ascii=False)
print(f"Wrote potential wrong-merge outputs: {wrong_jsonl} and {wrong_json}  (flagged groups: {len(wrong_merges)})")

# ------------------ Optional: Save clusters_processed copy ------------------
if clusters_processed:
    cp_out = OUT_DIR / "clusters_processed_copy.jsonl"
    with open(cp_out, "w", encoding="utf-8") as fh:
        for c in clusters_processed:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Wrote clusters_processed copy:", cp_out)

# ------------------ Final console summary ------------------
print("\n=== QUICK INSPECTION ===")
print("original mentions:", len(entities_raw))
print("resolved groups:", len(merged_groups_sorted))
print("near-miss group pairs found:", len(group_pairs_sorted))
print("near-miss mention pairs found:", len(pairs_sorted))
print("merge suggestions:", len(suggestions_sorted))
print("potential wrong-merged groups flagged:", len(wrong_merges))
print("\nTop 10 resolved groups (label -- size -- resolved_id):")
for g in merged_groups_sorted[:10]:
    print(f"  {g['label'][:70]:70s}  size={g['size']:3d}  id={g['resolved_id']}")
print("\nFiles written to:", OUT_DIR)
print(" - resolved_entity_table.csv")
print(" - merged_groups_full.json(.jsonl)")
print(" - merged_groups_by_size.csv")
print(" - singletons_unresolved.csv")
print(" - merged_groups_full_annotated.json(.jsonl/.csv)")
print(" - top_resolved_stats.json")
print(" - near_miss_pairs.csv")
print(" - near_miss_group_pairs.csv  (and _top.csv)")
print(" - merge_suggestions.jsonl")
print(" - wrong_merges.jsonl")
print("========================\n")



#endregion#? Analysis V3 - for merged resolution
#?#########################  End  ##########################












#?######################### Start ##########################
#region:#?   v4

#!/usr/bin/env python3
"""
analyze_entity_resolution_combined_with_nearmiss.py

Same as previous analysis script, but adds "near-miss" detection:
 - identifies mention-level pairs across different resolved groups whose
   textual similarity indicates they might deserve a merge (but weren't merged).
 - computes group-level similarity (representative labels) and suggests top group merges.
 - detects "potential wrong merges": resolved groups whose internal mean pairwise
   similarity is below NEAR_MISS_MIN (these may be incorrectly merged).

Outputs (in OUT_DIR):
 - near_miss_pairs.csv
 - near_miss_group_pairs.csv
 - merge_suggestions.jsonl
 - potential_wrong_merges.jsonl, potential_wrong_merges.csv
 + all previous analysis outputs (resolved_entity_table.csv, merged_groups_full.json, ...)

Notes:
 - Uses string-based similarity (difflib.SequenceMatcher + token Jaccard).
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import argparse
from difflib import SequenceMatcher
import itertools
import statistics
from typing import List, Dict, Any

# ------------------ Config / paths (change if needed) ------------------
# BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/entities_resolved_final.jsonl")
# ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")
# ENT_RESOLVED_JSONL = BASE_DIR / "entities_resolved_final.jsonl"
# RESOLVE_MAP = BASE_DIR / "resolve_map.json"
# HISTORY = BASE_DIR / "resolution_history.jsonl"
# CLUSTERS_PROCESSED = BASE_DIR / "clusters_processed.jsonl"
# CLUSTERS_FOR_REVIEW = BASE_DIR / "clusters_for_review.jsonl"
# LLM_DEBUG_DIR = BASE_DIR / "llm_debug"

# ------------------ Config / paths (change if needed) ------------------
BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved")
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw-entResTestssmaller.jsonl")

# NOTE: use the final two-stage outputs produced by your pipeline
OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENT_RESOLVED_JSONL = OUT_DIR / "entities_resolved_final.jsonl"
RESOLVE_MAP = OUT_DIR / "resolve_map_final.json"
HISTORY = OUT_DIR / "resolution_history_full.jsonl"

CLUSTERS_PROCESSED = BASE_DIR / "clusters_processed.jsonl"
CLUSTERS_FOR_REVIEW = BASE_DIR / "clusters_for_review.jsonl"
LLM_DEBUG_DIR = BASE_DIR / "llm_debug"




OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Near-miss thresholds (string-similarity based)
NEAR_MISS_MIN = 0.70   # lower bound to consider "close"
NEAR_MISS_MAX = 0.94   # if > this we may have expected a merge (tunable)
TOP_K_SUGGESTIONS = 200  # how many top candidates to write to suggestions

# ------------------ Helpers ------------------
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
            except Exception:
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"[WARN] failed to parse line in {path}: {ln[:120]!r}")
    return items

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(s.strip().lower().split())

def seq_ratio(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def token_jaccard(a: str, b: str) -> float:
    a_tokens = set([t for t in (a or "").lower().split() if t])
    b_tokens = set([t for t in (b or "").lower().split() if t])
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens.intersection(b_tokens)
    uni = a_tokens.union(b_tokens)
    return len(inter) / len(uni)

def composite_string_sim(name_a: str, name_b: str, desc_a: str, desc_b: str, weights=None) -> float:
    """
    Lightweight composite similarity using sequence ratio and token jaccard on names and descriptions.
    weights default tuned to prefer name similarity.
    """
    if weights is None:
        weights = {"name_seq":0.5, "name_jac":0.25, "desc_seq":0.15, "desc_jac":0.10}
    n_seq = seq_ratio(name_a, name_b)
    n_jac = token_jaccard(name_a, name_b)
    d_seq = seq_ratio(desc_a, desc_b)
    d_jac = token_jaccard(desc_a or "", desc_b or "")
    score = (n_seq * weights["name_seq"] + n_jac * weights["name_jac"] +
             d_seq * weights["desc_seq"] + d_jac * weights["desc_jac"])
    return float(score)

# ------------------ Load inputs ------------------
print("Loading input files (defaults shown in script)...")
entities_raw = load_jsonl(ENT_RAW)
print(f" - original mentions: {len(entities_raw)}  (ENT_RAW: {ENT_RAW})")

resolved_records = load_jsonl(ENT_RESOLVED_JSONL)
print(f" - canonical resolved records: {len(resolved_records)}  (ENT_RESOLVED_JSONL: {ENT_RESOLVED_JSONL})")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f" - resolve_map entries: {len(resolve_map)}  (RESOLVE_MAP: {RESOLVE_MAP})")

history = load_jsonl(HISTORY)
print(f" - history entries: {len(history)}  (HISTORY: {HISTORY})")

clusters_processed = load_jsonl(CLUSTERS_PROCESSED)
if clusters_processed:
    print(f" - clusters_processed entries: {len(clusters_processed)}  (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")
else:
    print(f" - clusters_processed missing or empty (CLUSTERS_PROCESSED: {CLUSTERS_PROCESSED})")

clusters_for_review = load_jsonl(CLUSTERS_FOR_REVIEW)
if clusters_for_review:
    print(f" - clusters_for_review entries: {len(clusters_for_review)}  (CLUSTERS_FOR_REVIEW: {CLUSTERS_FOR_REVIEW})")

entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_newid = { r.get("id_final"): r for r in resolved_records }

for r in resolved_by_newid.values():
    r.setdefault("label", r.get("label") or r.get("id_final"))
    r.setdefault("aliases", r.get("aliases") or [])
    r.setdefault("members", r.get("members") or [])
    r.setdefault("description", r.get("description") or "")

# ------------------ Existing analysis (unchanged) ------------------
csv_path = OUT_DIR / "resolved_entity_table.csv"
fields = [
    "original_id", "resolved_id", "resolved_label", "resolved_size", "resolved_aliases",
    "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence", "member_confidence"
]
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for ent in entities_raw:
        eid = ent.get("id")
        rid = resolve_map.get(eid)
        rlabel = ""
        rsize = ""
        raliases = ""
        member_conf = None
        if rid and rid in resolved_by_newid:
            rec = resolved_by_newid[rid]
            rlabel = rec.get("label","")
            rsize = len(rec.get("members",[]))
            raliases = "|".join(rec.get("aliases",[]))
            member_conf = rec.get("member_confidence", {}).get(eid) if isinstance(rec.get("member_confidence", {}), dict) else None
        row = {
            "original_id": eid,
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases,
            "entity_name": ent.get("entity_name",""),
            "entity_description": ent.get("entity_description",""),
            "entity_type_hint": ent.get("entity_type_hint",""),
            "chunk_id": ent.get("chunk_id") or ent.get("source_chunk") or ent.get("chunk") or "",
            "mention_confidence": ent.get("confidence_score") if ent.get("confidence_score") is not None else "",
            "member_confidence": member_conf if member_conf is not None else ""
        }
        writer.writerow(row)
print("Wrote:", csv_path)

# Build groups
groups = defaultdict(list)
for orig_id in entities_by_id:
    rid = resolve_map.get(orig_id)
    if rid:
        groups[rid].append(orig_id)
    else:
        groups[f"UNRESOLVED_{orig_id}"].append(orig_id)

merged_groups = []
for rid, members in groups.items():
    resolved_rec = resolved_by_newid.get(rid)
    label = resolved_rec.get("label") if resolved_rec else None
    aliases = resolved_rec.get("aliases") if resolved_rec else []
    resolved_desc = resolved_rec.get("description") if resolved_rec else ""
    member_objs = []
    for mid in members:
        m = entities_by_id.get(mid, {})
        member_objs.append({
            "id": mid,
            "entity_name": m.get("entity_name",""),
            "entity_description": m.get("entity_description",""),
            "entity_type_hint": m.get("entity_type_hint",""),
            "chunk_id": m.get("chunk_id") or m.get("source_chunk") or m.get("chunk") or "",
            "mention_confidence": m.get("confidence_score"),
            "member_confidence": (resolved_by_newid.get(rid, {}).get("member_confidence", {}).get(mid) if rid in resolved_by_newid else None)
        })
    merged_groups.append({
        "resolved_id": rid,
        "label": label or f"UNRESOLVED_GROUP_{rid}",
        "aliases": aliases,
        "members": member_objs,
        "size": len(member_objs),
        "resolved_description": resolved_desc
    })

merged_groups_sorted = sorted(merged_groups, key=lambda x: x["size"], reverse=True)

out_json = OUT_DIR / "merged_groups_full.json"
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(merged_groups_sorted, fh, indent=2, ensure_ascii=False)
out_jsonl = OUT_DIR / "merged_groups_full.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as fh:
    for g in merged_groups_sorted:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
print("Wrote merged groups files.")

csv_groups = OUT_DIR / "merged_groups_by_size.csv"
with open(csv_groups, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "aliases", "sample_member_names"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        sample_names = "; ".join([m.get("entity_name","") for m in g["members"][:6]])
        writer.writerow({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "aliases": "|".join(g.get("aliases",[])),
            "sample_member_names": sample_names
        })
print("Wrote:", csv_groups)

singletons_path = OUT_DIR / "singletons_unresolved.csv"
with open(singletons_path, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "mention_confidence"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for g in merged_groups_sorted:
        if g["size"] == 1 and (str(g["resolved_id"]).startswith("UNRESOLVED_") or g["label"].startswith("UNRESOLVED_")):
            m = g["members"][0]
            writer.writerow({
                "original_id": m.get("id"),
                "entity_name": m.get("entity_name",""),
                "entity_description": m.get("entity_description",""),
                "entity_type_hint": m.get("entity_type_hint",""),
                "chunk_id": m.get("chunk_id",""),
                "mention_confidence": m.get("mention_confidence","")
            })
print("Wrote:", singletons_path)

# ------------------ Annotate merge sources (unchanged) ------------------
entity_sources = defaultdict(list)
for h in history:
    act = (h.get("action") or "").lower()
    if act == "auto_merge":
        mids = h.get("member_ids") or h.get("merged_ids") or h.get("members") or []
        for m in mids:
            entity_sources[m].append("auto")
    elif act == "merge_entities":
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act in ("applied_merge_group", "applied_merge"):
        mids = h.get("member_ids") or h.get("merged_ids") or []
        for m in mids:
            entity_sources[m].append("llm")
    elif act == "rename_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    elif act == "kept_singleton":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("none")
    elif act == "left_unchanged_low_sim":
        for m in h.get("members", []) or []:
            entity_sources[m].append("none")
    elif act == "keep_entity":
        eid = h.get("entity_id")
        if eid:
            entity_sources[eid].append("llm")
    else:
        mids = h.get("merged_ids") or h.get("member_ids") or []
        for m in mids:
            entity_sources[m].append("llm" if act and "auto" not in act else "auto")

annotated = []
csv_rows = []
for g in merged_groups_sorted:
    members = g["members"]
    src_counter = Counter()
    for m in members:
        mid = m["id"]
        for src in entity_sources.get(mid, []):
            src_counter[src] += 1
    if not src_counter:
        resolution_source = "singleton"
    elif len(src_counter) == 1:
        resolution_source = next(iter(src_counter))
    else:
        resolution_source = "mixed"
    g2 = dict(g)
    g2["resolution_source"] = resolution_source
    g2["auto_merge_count"] = src_counter.get("auto", 0)
    g2["llm_merge_count"] = src_counter.get("llm", 0)
    annotated.append(g2)
    csv_rows.append({
        "resolved_id": g2["resolved_id"],
        "label": g2["label"],
        "size": g2["size"],
        "resolution_source": resolution_source,
        "auto_merge_count": g2["auto_merge_count"],
        "llm_merge_count": g2["llm_merge_count"],
        "sample_members": "; ".join(m["entity_name"] for m in members[:5])
    })

OUT_ANNOT_JSON = OUT_DIR / "merged_groups_full_annotated.json"
OUT_ANNOT_JSONL = OUT_DIR / "merged_groups_full_annotated.jsonl"
OUT_ANNOT_CSV = OUT_DIR / "merged_groups_by_source.csv"

with open(OUT_ANNOT_JSON, "w", encoding="utf-8") as fh:
    json.dump(annotated, fh, indent=2, ensure_ascii=False)
with open(OUT_ANNOT_JSONL, "w", encoding="utf-8") as fh:
    for g in annotated:
        fh.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(OUT_ANNOT_CSV, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["resolved_id", "label", "size", "resolution_source", "auto_merge_count", "llm_merge_count", "sample_members"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)
print("Wrote annotated merged groups.")

# ------------------ Stats (unchanged) ------------------
size_counts = Counter([g["size"] for g in merged_groups_sorted])
top_labels = [{"label":g["label"], "size":g["size"], "resolved_id": g["resolved_id"]} for g in merged_groups_sorted[:50]]
type_dist = Counter([ e.get("entity_type_hint") or "None" for e in entities_raw ])
confidence_vals = []
for e in entities_raw:
    cs = e.get("confidence_score")
    try:
        if cs is not None:
            confidence_vals.append(float(cs))
    except Exception:
        pass

conf_summary = {}
if confidence_vals:
    conf_summary = {
        "count": len(confidence_vals),
        "mean": sum(confidence_vals)/len(confidence_vals),
        "min": min(confidence_vals),
        "max": max(confidence_vals)
    }

stats = {
    "n_original_mentions": len(entities_raw),
    "n_resolved_groups": len(merged_groups_sorted),
    "size_distribution": dict(size_counts),
    "top_resolved_samples": top_labels,
    "type_distribution_sample": type_dist.most_common(40),
    "confidence_summary": conf_summary,
    "history_action_counts": Counter([h.get("action") for h in history])
}

stats_out = OUT_DIR / "top_resolved_stats.json"
with open(stats_out, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
print("Wrote:", stats_out)

# ------------------ NEW: Near-miss detection ------------------

# Build representative label/desc for each group:
group_reprs = {}
for g in merged_groups_sorted:
    rid = g["resolved_id"]
    label = g.get("label") or ""
    desc = g.get("resolved_description") or ""
    # fallback to first member
    if (not label or label.startswith("UNRESOLVED_")) and g["members"]:
        label = label if label and not label.startswith("UNRESOLVED_") else (g["members"][0].get("entity_name") or "")
    if (not desc) and g["members"]:
        desc = desc or (g["members"][0].get("entity_description") or "")
    group_reprs[rid] = {"label": str(label), "desc": str(desc), "size": g["size"], "members": [m["id"] for m in g["members"]]}

# 1) Group-level similarity (representative labels)
group_pairs = []
group_ids = list(group_reprs.keys())
for a, b in itertools.combinations(group_ids, 2):
    la = group_reprs[a]["label"]
    lb = group_reprs[b]["label"]
    da = group_reprs[a]["desc"]
    db = group_reprs[b]["desc"]
    score = composite_string_sim(la, lb, da, db)
    if score >= NEAR_MISS_MIN:
        group_pairs.append((a, b, score))

# sort and write group-level near-misses
group_pairs_sorted = sorted(group_pairs, key=lambda x: x[2], reverse=True)
gp_csv = OUT_DIR / "near_miss_group_pairs.csv"
with open(gp_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a", "group_b", "label_a", "label_b", "size_a", "size_b", "score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"],
            "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"],
            "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote group-level near-miss csv:", gp_csv)

# 2) Mention-level cross-group near-miss pairs (top K)
pairs = []
mention_items = []
# prepare mention list with resolved group id
for ent in entities_raw:
    mid = ent.get("id")
    rid = resolve_map.get(mid) or f"UNRES_{mid}"
    mention_items.append({
        "id": mid,
        "rid": rid,
        "name": ent.get("entity_name","") or "",
        "desc": ent.get("entity_description","") or ""
    })

# For efficiency, compare only candidate group pairs (from group_pairs_sorted). If none, fallback to top groups.
candidate_group_pairs = [ (a,b) for (a,b,sc) in group_pairs_sorted ]
if not candidate_group_pairs:
    top_groups = [g["resolved_id"] for g in merged_groups_sorted[:50]]
    candidate_group_pairs = list(itertools.combinations(top_groups, 2))

# build map group->members (mention dicts)
group_to_mentions = defaultdict(list)
for m in mention_items:
    group_to_mentions[m["rid"]].append(m)

# compute mention-level scores for candidate group pairs
for a,b in candidate_group_pairs:
    membs_a = group_to_mentions.get(a, [])
    membs_b = group_to_mentions.get(b, [])
    # skip trivial empty groups
    if not membs_a or not membs_b:
        continue
    # cap per-group comparisons to avoid explosion
    cap = 40
    membs_a_sample = membs_a[:cap]
    membs_b_sample = membs_b[:cap]
    for ma in membs_a_sample:
        for mb in membs_b_sample:
            name_score = seq_ratio(ma["name"], mb["name"])
            comp_score = composite_string_sim(ma["name"], mb["name"], ma["desc"], mb["desc"])
            if comp_score >= NEAR_MISS_MIN:
                pairs.append({
                    "mention_a": ma["id"], "mention_b": mb["id"],
                    "group_a": a, "group_b": b,
                    "name_a": ma["name"], "name_b": mb["name"],
                    "desc_a": ma["desc"], "desc_b": mb["desc"],
                    "score": comp_score,
                    "name_ratio": name_score
                })

pairs_filtered = [p for p in pairs if p["score"] >= NEAR_MISS_MIN]
pairs_sorted = sorted(pairs_filtered, key=lambda x: x["score"], reverse=True)

pairs_csv = OUT_DIR / "near_miss_pairs.csv"
with open(pairs_csv, "w", encoding="utf-8", newline="") as fh:
    fieldnames = ["mention_a","mention_b","group_a","group_b","name_a","name_b","score","name_ratio"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for p in pairs_sorted:
        writer.writerow({
            "mention_a": p["mention_a"],
            "mention_b": p["mention_b"],
            "group_a": p["group_a"],
            "group_b": p["group_b"],
            "name_a": p["name_a"],
            "name_b": p["name_b"],
            "score": f"{p['score']:.4f}",
            "name_ratio": f"{p['name_ratio']:.4f}"
        })
print("Wrote mention-level near-miss pairs:", pairs_csv)

# 3) Merge suggestions: aggregate group-level + mention-level evidence
group_evidence = defaultdict(list)
for p in pairs_sorted:
    key = tuple(sorted([p["group_a"], p["group_b"]]))
    group_evidence[key].append(p["score"])

suggestions = []
for (a,b,sc) in group_pairs_sorted:
    key = tuple(sorted([a,b]))
    mention_scores = group_evidence.get(key, [])
    best_m = max(mention_scores) if mention_scores else 0.0
    combined = sc * 0.6 + best_m * 0.4
    suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": sc, "best_mention_score": best_m, "combined_score": combined})

# also include group pairs that only came from mention evidence
for key, mention_scores in group_evidence.items():
    if not any((key[0]==g[0] and key[1]==g[1]) or (key[0]==g[1] and key[1]==g[0]) for g in [(gp[0],gp[1]) for gp in group_pairs_sorted]):
        best_m = max(mention_scores)
        combined = best_m
        suggestions.append({"group_a": key[0], "group_b": key[1], "group_score": None, "best_mention_score": best_m, "combined_score": combined})

suggestions_sorted = sorted(suggestions, key=lambda x: (x["combined_score"] or 0.0), reverse=True)[:TOP_K_SUGGESTIONS]
sugg_out = OUT_DIR / "merge_suggestions.jsonl"
with open(sugg_out, "w", encoding="utf-8") as fh:
    for s in suggestions_sorted:
        fh.write(json.dumps(s, ensure_ascii=False) + "\n")
print("Wrote merge suggestions:", sugg_out)

gp_top_csv = OUT_DIR / "near_miss_group_pairs_top.csv"
with open(gp_top_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["group_a","group_b","label_a","label_b","size_a","size_b","score"])
    writer.writeheader()
    for a,b,sc in group_pairs_sorted[:TOP_K_SUGGESTIONS]:
        writer.writerow({
            "group_a": a, "group_b": b,
            "label_a": group_reprs[a]["label"], "label_b": group_reprs[b]["label"],
            "size_a": group_reprs[a]["size"], "size_b": group_reprs[b]["size"],
            "score": f"{sc:.4f}"
        })
print("Wrote top group-level near-miss csv:", gp_top_csv)

# ------------------ NEW: Potential wrong merges (internal group coherence) ------------------
potential_wrong = []
for g in merged_groups_sorted:
    if g["size"] <= 1:
        continue
    # compute mean pairwise similarity among members (names+descs)
    member_pairs = []
    members = g["members"]
    for a_idx in range(len(members)):
        for b_idx in range(a_idx+1, len(members)):
            ma = members[a_idx]
            mb = members[b_idx]
            sc = composite_string_sim(ma.get("entity_name",""), mb.get("entity_name",""),
                                      ma.get("entity_description",""), mb.get("entity_description",""))
            member_pairs.append(sc)
    mean_internal = float(statistics.mean(member_pairs)) if member_pairs else 0.0
    if mean_internal < NEAR_MISS_MIN:
        potential_wrong.append({
            "resolved_id": g["resolved_id"],
            "label": g["label"],
            "size": g["size"],
            "mean_internal_member_similarity": mean_internal,
            "sample_members": [m.get("entity_name","") for m in members[:8]]
        })

# write potential wrong merges jsonl & csv
pw_jsonl = OUT_DIR / "potential_wrong_merges.jsonl"
with open(pw_jsonl, "w", encoding="utf-8") as fh:
    for p in potential_wrong:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")
pw_csv = OUT_DIR / "potential_wrong_merges.csv"
with open(pw_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["resolved_id","label","size","mean_internal_member_similarity","sample_members"])
    writer.writeheader()
    for p in potential_wrong:
        writer.writerow({
            "resolved_id": p["resolved_id"],
            "label": p["label"],
            "size": p["size"],
            "mean_internal_member_similarity": f"{p['mean_internal_member_similarity']:.4f}",
            "sample_members": " | ".join(p["sample_members"])
        })
print("Wrote potential wrong merges:", pw_jsonl, pw_csv)

# ------------------ Optional: Save clusters_processed copy ------------------
if clusters_processed:
    cp_out = OUT_DIR / "clusters_processed_copy.jsonl"
    with open(cp_out, "w", encoding="utf-8") as fh:
        for c in clusters_processed:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Wrote clusters_processed copy:", cp_out)

# ------------------ Final console summary ------------------
print("\n=== QUICK INSPECTION ===")
print("original mentions:", len(entities_raw))
print("resolved groups:", len(merged_groups_sorted))
print("potential wrong merges (count):", len(potential_wrong))
print("top 10 resolved groups (label -- size -- resolved_id):")
for g in merged_groups_sorted[:10]:
    print(f"  {g['label'][:70]:70s}  size={g['size']:3d}  id={g['resolved_id']}")
print("\nFiles written to:", OUT_DIR)
print(" - resolved_entity_table.csv")
print(" - merged_groups_full.json(.jsonl)")
print(" - merged_groups_by_size.csv")
print(" - singletons_unresolved.csv")
print(" - merged_groups_full_annotated.json(.jsonl/.csv)")
print(" - top_resolved_stats.json")
print(" - near_miss_pairs.csv")
print(" - near_miss_group_pairs.csv  (and _top.csv)")
print(" - merge_suggestions.jsonl")
print(" - potential_wrong_merges.jsonl / .csv")
if clusters_for_review:
    print(" - clusters_flagged_for_review.jsonl")
if LLM_DEBUG_DIR.exists():
    print(" - LLM debug dir (exists):", LLM_DEBUG_DIR)
print("========================\n")



#endregion#? V4
#?#########################  End  ##########################












#?######################### Start ##########################
#region:#?   Entity Resolution V0

#endregion#? Entity Resolution V0
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Entity Resolution V


#endregion#? Entity Resolution V
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Entity Resolution V


#endregion#? Entity Resolution V
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   Entity Resolution V


#endregion#? Entity Resolution V
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











