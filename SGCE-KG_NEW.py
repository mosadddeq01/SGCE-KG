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




