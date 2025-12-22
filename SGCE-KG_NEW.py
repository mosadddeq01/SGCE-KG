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
#region:#?   Clustetting playgound 1

#!/usr/bin/env python3


"""
embed_and_viz_entities.py

Reads entities from a jsonl file (entities_raw.jsonl), computes embeddings using
selected models, reduces dimensionality, saves CSVs and PNGs for inspection.

Usage examples:
 python embed_and_viz_entities.py --model hf_all-MiniLM-L6-v2
 python embed_and_viz_entities.py --model openai_text-embedding-3-small
 python embed_and_viz_entities.py --model tfidf
 python embed_and_viz_entities.py --model hf_all-MiniLM-L6-v2 --reduce umap
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports (wrapped to provide helpful message)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

try:
    import umap
except Exception:
    umap = None

# ---------- CONFIG ----------
ENT_FILE_DEFAULT = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl"
OUT_DIR = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entity_embedding_outputs"
# ----------------------------

def load_entities(jsonl_path):
    ents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # ensure keys exist
            ents.append(obj)
    return ents

def build_text_for_entity(ent, max_ctx_snippets=2):
    """
    Build a representative text for embedding from entity fields.
    """
    parts = []
    if ent.get("entity_name"):
        parts.append(ent["entity_name"])
    desc = ent.get("entity_description") or ""
    if desc.strip():
        parts.append(desc.strip())
    # If used_context_ids present and your chunk DB is not accessible here,
    # include text_span and other metadata to give local context
    if ent.get("text_span"):
        parts.append(ent["text_span"])
    # add type hint for disambiguation
    if ent.get("entity_type_hint"):
        parts.append(f"TYPE:{ent['entity_type_hint']}")
    return " ||| ".join([p for p in parts if p])

def embed_with_sentence_transformer(model_name, texts, device="cpu", batch_size=64):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embeddings

def embed_with_openai(model_name, texts, batch_size=50):
    if openai is None:
        raise RuntimeError("openai package not installed. `pip install openai`")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    openai.api_key = key

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI embed batches"):
        batch = texts[i:i+batch_size]
        # openai API: responses for multiple inputs
        resp = openai.Embedding.create(model=model_name, input=batch)
        for r in resp["data"]:
            embeddings.append(np.array(r["embedding"], dtype=np.float32))
    return np.vstack(embeddings)

def embed_with_tfidf(texts, n_components=768):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X = vec.fit_transform(texts)  # sparse
    # reduce to dense lower dim via PCA if n_components < features
    dense = X.toarray()
    if n_components and dense.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=0)
        dense = pca.fit_transform(dense)
    return dense

def reduce_dimensions(embs, method="umap", n_components=2, pca_first=True, pca_dim=50, random_state=42):
    """
    pca_first: reduces to pca_dim before applying UMAP/t-SNE (good for speed/stability)
    method: 'umap' | 'tsne' | 'pca'
    """
    X = embs
    if pca_first and method in ("umap", "tsne") and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X = pca.fit_transform(X)
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(X)
    elif method == "tsne":
        ts = TSNE(n_components=n_components, random_state=random_state, perplexity=30, n_iter=1000, init="pca")
        return ts.fit_transform(X)
    elif method == "umap":
        if umap is None:
            raise RuntimeError("umap-learn not installed. `pip install umap-learn`")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(X)
    else:
        raise ValueError("Unknown reduction method")

def plot_scatter(coords, labels, title, out_path, sample_ids=None, figsize=(10,8), annotate=False, max_annot=50):
    df = pd.DataFrame(coords, columns=["x","y"])
    df["label"] = labels
    plt.figure(figsize=figsize)
    uniq = list(set(labels))
    # color by label if few labels, else grey
    if len(uniq) <= 20:
        for lbl in uniq:
            mask = df["label"] == lbl
            plt.scatter(df.loc[mask, "x"], df.loc[mask, "y"], label=str(lbl), alpha=0.7, s=20)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        plt.scatter(df["x"], df["y"], alpha=0.7, s=18)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    # annotate a few
    if annotate:
        import random
        n = min(max_annot, len(df))
        idxs = random.sample(range(len(df)), n)
        for i in idxs:
            plt.annotate(sample_ids[i] if sample_ids is not None else str(i),
                         (coords[i,0], coords[i,1]), fontsize=6)
    plt.close()

def main(args):
    ent_file = Path(args.entities)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading entities...")
    ents = load_entities(ent_file)
    print(f"Loaded {len(ents)} entities.")

    # build texts
    texts = [build_text_for_entity(e) for e in ents]
    ids = [e.get("id") for e in ents]
    names = [e.get("entity_name","") for e in ents]
    types = [e.get("entity_type_hint","") for e in ents]

    # Optionally subsample for quick tests
    if args.limit and args.limit < len(texts):
        texts = texts[:args.limit]
        ids = ids[:args.limit]
        names = names[:args.limit]
        types = types[:args.limit]

    # Compute embeddings
    model = args.model
    print("Embedding model:", model)
    if model.startswith("hf_"):
        hf_model = model[len("hf_"):]
        emb = embed_with_sentence_transformer(hf_model, texts, device=args.device, batch_size=args.batch_size)
    elif model.startswith("openai_"):
        if openai is None:
            raise RuntimeError("openai package not available.")
        openai_model = model[len("openai_"):]
        emb = embed_with_openai(openai_model, texts, batch_size=args.batch_size)
    elif model == "tfidf":
        emb = embed_with_tfidf(texts, n_components=args.tfidf_dim)
    else:
        raise ValueError("Unknown model format. Use hf_<model>, openai_<model>, or tfidf")

    print("Embeddings shape:", emb.shape)

    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb = emb / norms

    # Save raw embeddings
    emb_df = pd.DataFrame(emb)
    emb_df.insert(0, "id", ids[:len(emb)])
    emb_df["name"] = names[:len(emb)]
    emb_df["type"] = types[:len(emb)]
    emb_df.to_csv(outdir / f"embeddings_{model}.csv", index=False)
    print("Saved embeddings CSV.")

    # Dimensionality reduction
    print("Reducing dimensions...")
    coords = reduce_dimensions(emb, method=args.reduction, n_components=2,
                               pca_first=not args.no_pca_first, pca_dim=args.pca_first_dim, random_state=args.seed)
    print("Reduced shape:", coords.shape)

    # Save coords
    coords_df = pd.DataFrame(coords, columns=["x","y"])
    coords_df["id"] = ids[:len(coords)]
    coords_df["name"] = names[:len(coords)]
    coords_df["type"] = types[:len(coords)]
    coords_df.to_csv(outdir / f"coords_{model}_{args.reduction}.csv", index=False)

    # Plot
    title = f"{model} -> {args.reduction}"
    plot_path = outdir / f"scatter_{model}_{args.reduction}.png"
    plot_scatter(coords, types, title, plot_path, sample_ids=ids, annotate=args.annotate)
    print("Saved scatter plot to", plot_path)

    print("Done.")

class Args:
    entities = ENT_FILE_DEFAULT
    model = "hf_all-MiniLM-L6-v2"
    reduction = "umap"
    outdir = OUT_DIR
    limit = 500
    device = "cpu"
    batch_size = 64
    tfidf_dim = 512
    no_pca_first = False
    pca_first_dim = 50
    seed = 42
    annotate = True

args = Args()
main(args)

#endregion#? Clustetting playgound
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Clustering playgound 2 - Building a report

"""
entity_stats.py

Compute diagnostics & summary statistics over provisional entities.

Outputs:
 - prints a short human summary to stdout
 - writes `entity_stats_report.json` with detailed metrics
 - writes CSVs: name_frequencies.csv, entities_by_chunk.csv
 - optional embedding-based outputs (requires embeddings .npy and faiss/hdbscan installed)

Usage:
    python entity_stats.py

Dependencies:
    pip install pandas numpy tqdm faiss-cpu scikit-learn hdbscan

If you don't have faiss/hdbscan, the script will still run lexical + confidence stats.
"""

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional imports
try:
    import faiss
except Exception:
    faiss = None

try:
    import hdbscan
except Exception:
    hdbscan = None

# --------- Config (edit paths if necessary) ----------
ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
EMBEDDINGS_NPY = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb.npy")   # optional: per-entity embeddings (order must match entities order)
EMBED_META_NDX = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb_meta.jsonl")  # optional: mapping between embeddings rows and entity ids
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entity_stats_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = OUT_DIR / "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entity_stats_report.json"

# --------- Utility helpers ----------
def load_entities(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                # be forgiving: try to fix trailing commas or single quotes (if simple)
                try:
                    e = json.loads(line.replace("'", '"'))
                except Exception:
                    continue
            ents.append(e)
    return ents

def normalize_name(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s2 = s.strip().lower()
    # remove common version tokens, punctuation, extra whitespace
    s2 = s2.replace("_", " ")
    s2 = s2.replace("-", " ")
    s2 = s2.replace(".", " ")
    # strip typical version markers like v1, v2, 1.0, 2.0
    s2 = __import__("re").sub(r"\b(v|version)\s*\d+(\.\d+)?\b", "", s2)
    s2 = __import__("re").sub(r"\s+", " ", s2).strip()
    return s2

# --------- Core stats functions ----------
def lexical_stats(entities: List[Dict]) -> Dict:
    total = len(entities)
    names = [ (e.get("entity_name") or e.get("name") or "").strip() for e in entities ]
    normalized = [ normalize_name(n) for n in names ]
    non_empty_names = [n for n in names if n]
    non_empty_norm = [n for n in normalized if n]
    uniq_names = len(set(non_empty_names))
    uniq_norm = len(set(non_empty_norm))

    name_counts = Counter(non_empty_names)
    norm_counts = Counter(non_empty_norm)

    top_names = name_counts.most_common(30)
    top_norm = norm_counts.most_common(30)

    # how many singletons vs repeated
    singleton_count = sum(1 for v in norm_counts.values() if v == 1)
    repeated_count = sum(1 for v in norm_counts.values() if v > 1)

    return {
        "total_entities": total,
        "non_empty_name_entities": len(non_empty_names),
        "unique_surface_names": uniq_names,
        "unique_normalized_names": uniq_norm,
        "top_surface_names": top_names,
        "top_normalized_names": top_norm,
        "singleton_normalized_count": singleton_count,
        "repeated_normalized_count": repeated_count,
    }

def type_and_confidence_stats(entities: List[Dict]) -> Dict:
    type_counter = Counter()
    confs = []
    conf_present = 0
    for e in entities:
        t = e.get("entity_type_hint") or e.get("type") or "Unknown"
        type_counter[t] += 1
        c = e.get("confidence_score")
        try:
            if c is not None:
                confs.append(float(c))
                conf_present += 1
        except Exception:
            continue

    conf_summary = None
    if confs:
        arr = np.array(confs)
        conf_summary = {
            "count": len(confs),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    return {
        "type_distribution": type_counter.most_common(50),
        "confidence_summary": conf_summary,
        "entities_with_confidence": conf_present
    }

def per_chunk_stats(entities: List[Dict]) -> Dict:
    by_chunk = defaultdict(list)
    for e in entities:
        chunk = e.get("chunk_id") or e.get("source_chunk") or e.get("source_chunks",[None])[0]
        by_chunk[chunk].append(e)
    sizes = [ len(v) for v in by_chunk.values() ]
    sizes_arr = np.array(sizes) if sizes else np.array([])
    summary = {}
    if sizes_arr.size:
        summary = {
            "n_chunks_with_entities": int(len(sizes)),
            "mean_entities_per_chunk": float(sizes_arr.mean()),
            "median_entities_per_chunk": float(np.median(sizes_arr)),
            "max_entities_in_chunk": int(sizes_arr.max()),
            "min_entities_in_chunk": int(sizes_arr.min()),
            "p90": float(np.percentile(sizes_arr, 90))
        }
    else:
        summary = {
            "n_chunks_with_entities": 0
        }
    # also produce top chunks
    top_chunks = sorted(by_chunk.items(), key=lambda kv: len(kv[1]), reverse=True)[:30]
    top_chunks_summary = [ (chunk, len(lst)) for chunk,lst in top_chunks ]
    summary["top_chunks"] = top_chunks_summary
    return summary, by_chunk

# --------- Embedding-based helpers (optional) ----------
def load_embeddings(emb_path: Path, meta_path: Optional[Path]=None):
    if not emb_path.exists():
        return None, None
    vecs = np.load(str(emb_path))
    meta = None
    if meta_path and meta_path.exists():
        meta = [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]
    return vecs, meta

def build_faiss_index(vecs: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss not installed")
    dim = vecs.shape[1]
    # use inner-product on normalized vectors to get cosine
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx

def embedding_density_stats(vecs: np.ndarray, idx, topN: int = 64) -> Dict:
    """
    For each vector, query topN neighbors and compute:
      - neighbor count above thresholds
      - mean/top similarity distribution
    Returns aggregate summaries.
    """
    if vecs is None or idx is None:
        return {}
    n = vecs.shape[0]
    sims_topk = []
    counts_above = defaultdict(int)
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75]
    for i in tqdm(range(n), desc="FAISS neighbor stats"):
        q = vecs[i:i+1]
        D, I = idx.search(q, topN+1)  # includes self
        sims = D[0].tolist()
        # drop self (should be 1.0)
        sims_no_self = [s for s in sims if s < 0.9999]
        if not sims_no_self:
            sims_topk.append([0.0])
            continue
        sims_topk.append(sims_no_self[:topN])
        for th in thresholds:
            if any(s >= th for s in sims_no_self):
                counts_above[th] += 1
    # compute aggregated metrics
    avg_top1 = float(np.mean([s[0] if s else 0.0 for s in sims_topk]))
    p25 = float(np.percentile([s[0] if s else 0.0 for s in sims_topk], 25))
    p75 = float(np.percentile([s[0] if s else 0.0 for s in sims_topk], 75))
    return {
        "n_vectors": int(n),
        "avg_top1_similarity": avg_top1,
        "top1_p25": p25,
        "top1_p75": p75,
        "counts_above_thresholds": { str(k): int(v) for k,v in counts_above.items() }
    }

def run_hdbscan_clustering(vecs: np.ndarray, min_cluster_size: int = 3):
    if hdbscan is None:
        return None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(vecs)
    # label -1 are noise/outliers
    counts = Counter(labels)
    return {
        "labels": labels,
        "cluster_counts": counts.most_common()
    }

# --------- Main driver ----------
def build_report():
    ents = load_entities(ENTITIES_PATH)
    print(f"Loaded {len(ents)} entities from {ENTITIES_PATH}")

    # Basic lexical / name stats
    lex = lexical_stats(ents)
    types = type_and_confidence_stats(ents)
    chunk_summary, by_chunk = per_chunk_stats(ents)

    # Save some CSVs for inspection
    # name freq
    name_rows = []
    for n,c in Counter([ (e.get("entity_name") or "").strip() for e in ents if (e.get("entity_name") or "").strip() ]).most_common():
        name_rows.append({"entity_name": n, "count": c, "normalized": normalize_name(n)})
    df_names = pd.DataFrame(name_rows)
    df_names.to_csv(OUT_DIR / "name_frequencies.csv", index=False)

    # entities by chunk
    chunk_rows = []
    for chunk_id, lst in by_chunk.items():
        for e in lst:
            chunk_rows.append({
                "chunk_id": chunk_id,
                "entity_id": e.get("id"),
                "entity_name": e.get("entity_name"),
                "entity_type_hint": e.get("entity_type_hint"),
                "confidence_score": e.get("confidence_score")
            })
    df_chunks = pd.DataFrame(chunk_rows)
    df_chunks.to_csv(OUT_DIR / "entities_by_chunk.csv", index=False)

    report = {
        "lexical": lex,
        "types": types,
        "per_chunk": chunk_summary,
        "n_entities": len(ents),
    }

    # Embedding-based analysis (if embeddings present)
    vecs, meta = load_embeddings(EMBEDDINGS_NPY, EMBED_META_NDX if EMBED_META_NDX.exists() else None)
    if vecs is not None:
        try:
            idx = build_faiss_index(vecs)
            emb_stats = embedding_density_stats(vecs, idx, topN=64)
            report["embedding_stats"] = emb_stats
        except Exception as e:
            report["embedding_stats_error"] = str(e)

        # optional clustering
        if hdbscan is not None:
            try:
                cluster_res = run_hdbscan_clustering(vecs, min_cluster_size=3)
                report["hdbscan_summary"] = {
                    "n_clusters_including_noise": len(set(cluster_res["labels"])),
                    "cluster_counts_top": cluster_res["cluster_counts"][:30]
                }
            except Exception as e:
                report["hdbscan_error"] = str(e)
        else:
            report["hdbscan_available"] = False
    else:
        report["embedding_stats"] = None
        report["hdbscan_available"] = hdbscan is not None

    # Write JSON report
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Print a short human-friendly summary
    print("\n=== ENTITY STATS SUMMARY ===")
    print(f"Total provisional entities: {report['n_entities']}")
    print(f"Unique surface names: {report['lexical']['unique_surface_names']}")
    print(f"Unique normalized names: {report['lexical']['unique_normalized_names']}")
    print(f"Singleton (normalized) names: {report['lexical']['singleton_normalized_count']}")
    print(f"Repeated (normalized) names: {report['lexical']['repeated_normalized_count']}")
    print(f"Top surface names saved to: {OUT_DIR / 'name_frequencies.csv'}")
    print(f"Entities-by-chunk saved to: {OUT_DIR / 'entities_by_chunk.csv'}")
    if report.get("types", None):
        print(f"Top entity types (sample): {report['types']['type_distribution'][:10]}")
    if report.get("embedding_stats"):
        print("Embedding stats (avg top1 similarity):", report["embedding_stats"]["avg_top1_similarity"])
    if report.get("hdbscan_summary"):
        print("HDBSCAN clusters (top):", report["hdbscan_summary"]["cluster_counts_top"][:10])
    print(f"Full JSON report saved to: {REPORT_JSON}")

if __name__ == "__main__":
    build_report()


#endregion#? Clustering playgound 2 - Building a report
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   

"""
entity_stats_with_clusters.py

Run entity diagnostics + embedding neighbor stats + HDBSCAN clustering
using the paths you provided.

Outputs (saved under OUT_DIR):
 - entity_stats_report.json
 - name_frequencies.csv
 - entities_by_chunk.csv
 - embedding_neighbor_stats.json
 - hdbscan_clusters.json
 - top_clusters_examples.csv
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional libs
try:
    import faiss
except Exception:
    faiss = None
try:
    import hdbscan
except Exception:
    hdbscan = None

# ---------------- CONFIG (from your message) ----------------
ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
EMBEDDINGS_NPY = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb.npy")
EMBED_META_NDX = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb_meta.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entity_stats_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = OUT_DIR / "entity_stats_report.json"

# Tunable params
TOPN_NEIGHBORS = 64
MIN_CLUSTER_SIZE = 3        # HDBSCAN min_cluster_size
EXEMPLAR_COUNT = 5          # per cluster exemplars to save
LARGE_CLUSTER_THRESH = 50   # threshold to call cluster "large"
SIM_THRESH_SUGGEST = 0.82   # suggested similarity threshold inferred from data

# ---------------- Helpers ----------------
def load_entities(p: Path) -> List[Dict]:
    if not p.exists():
        raise FileNotFoundError(f"Entities file not found: {p}")
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                # last-resort tolerant parse
                try:
                    out.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    continue
    return out

def normalize_name(s: str) -> str:
    if not s:
        return ""
    s2 = s.strip().lower()
    import re
    s2 = s2.replace("_", " ").replace("-", " ").replace(".", " ")
    s2 = re.sub(r"\b(v|version)\s*\d+(\.\d+)?\b", "", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def load_embeddings(emb_path: Path, meta_path: Optional[Path]=None):
    if not emb_path.exists():
        return None, None
    vecs = np.load(str(emb_path))
    meta = None
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = [json.loads(l) for l in fh]
    return vecs, meta

def build_faiss_index(vecs: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss is not installed or importable.")
    # normalize for cosine (inner product after normalization)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx

# ---------------- Stats functions ----------------
def basic_lexical_stats(entities: List[Dict]):
    total = len(entities)
    names = [ (e.get("entity_name") or e.get("name") or "").strip() for e in entities ]
    non_empty = [n for n in names if n]
    uniq_surface = len(set(non_empty))
    normals = [normalize_name(n) for n in non_empty]
    uniq_norm = len(set(normals))
    norm_counts = Counter(normals)
    top_norm = norm_counts.most_common(50)
    singleton_norm = sum(1 for v in norm_counts.values() if v == 1)
    repeated_norm = sum(1 for v in norm_counts.values() if v > 1)

    return {
        "total_entities": total,
        "non_empty_names": len(non_empty),
        "unique_surface_names": uniq_surface,
        "unique_normalized_names": uniq_norm,
        "top_normalized_names": top_norm[:50],
        "singleton_normalized_count": singleton_norm,
        "repeated_normalized_count": repeated_norm
    }

def type_confidence_stats(entities: List[Dict]):
    type_cnt = Counter()
    confs = []
    for e in entities:
        t = e.get("entity_type_hint") or e.get("type") or "Unknown"
        type_cnt[t] += 1
        c = e.get("confidence_score")
        try:
            if c is not None:
                confs.append(float(c))
        except Exception:
            pass
    conf_summary = None
    if confs:
        arr = np.array(confs)
        conf_summary = {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr,25)),
            "p75": float(np.percentile(arr,75))
        }
    return {"type_distribution": type_cnt.most_common(60), "confidence_summary": conf_summary}

def per_chunk_distribution(entities: List[Dict]):
    by_chunk = defaultdict(list)
    for e in entities:
        chunk = e.get("chunk_id") or e.get("source_chunk") or (e.get("source_chunks") and e.get("source_chunks")[0]) or None
        by_chunk[chunk].append(e)
    sizes = [len(v) for v in by_chunk.values()]
    import numpy as np
    if sizes:
        return {
            "n_chunks_with_entities": int(len(sizes)),
            "mean_entities_per_chunk": float(np.mean(sizes)),
            "median_entities_per_chunk": float(np.median(sizes)),
            "max_entities_in_chunk": int(max(sizes)),
            "min_entities_in_chunk": int(min(sizes)),
            "p90": float(np.percentile(sizes,90)),
            "top_chunks": sorted([(k,len(v)) for k,v in by_chunk.items()], key=lambda x: x[1], reverse=True)[:50]
        }, by_chunk
    else:
        return {"n_chunks_with_entities": 0}, by_chunk

# ---------------- Embedding neighbor diagnostics ----------------
def embedding_neighbor_stats(vecs: np.ndarray, idx, topN: int = 64):
    n = vecs.shape[0]
    sims_top1 = []
    counts_above = {0.95:0, 0.90:0, 0.85:0, 0.80:0, 0.75:0}
    for i in tqdm(range(n), desc="neighbor stats"):
        q = vecs[i:i+1]
        D,I = idx.search(q, topN+1)  # includes self
        sims = D[0].tolist()
        # drop self (near 1.0)
        sims_no_self = [s for s in sims if s < 0.9999]
        if len(sims_no_self)==0:
            sims_top1.append(0.0)
            continue
        top1 = sims_no_self[0]
        sims_top1.append(top1)
        for th in counts_above.keys():
            if any(s >= th for s in sims_no_self):
                counts_above[th] += 1
    import numpy as np
    arr = np.array(sims_top1)
    return {
        "n_vectors": int(n),
        "avg_top1_similarity": float(arr.mean()),
        "top1_p25": float(np.percentile(arr,25)),
        "top1_p50": float(np.percentile(arr,50)),
        "top1_p75": float(np.percentile(arr,75)),
        "counts_above": {str(k):int(v) for k,v in counts_above.items()}
    }

# ---------------- HDBSCAN clustering ----------------
def run_hdbscan(vecs: np.ndarray, min_cluster_size: int = MIN_CLUSTER_SIZE):
    if hdbscan is None:
        return None
    # Use euclidean on normalized vectors; alternatively use precomputed distances
    # If vectors are normalized for cosine (faiss.normalize_L2 was applied), euclidean works but metric could be 'cosine' too.
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True)
        labels = clusterer.fit_predict(vecs)
        counts = Counter(labels)
        return {"labels": labels.tolist(), "cluster_counts": counts.most_common(100)}
    except Exception as e:
        return {"error": str(e)}

# ---------------- cluster exemplars ----------------
def cluster_exemplars(labels: List[int], entities: List[Dict], vecs: np.ndarray, topk: int = EXEMPLAR_COUNT):
    # produce for each cluster: size, topk exemplar names (closest to centroid)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[lab].append(i)
    out = []
    for lab, idxs in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
        if lab == -1:
            # noise
            continue
        members = idxs
        size = len(members)
        # centroid (mean)
        centroid = np.mean(vecs[members], axis=0)
        # compute cosine sims (vecs assumed normalized)
        sims = (vecs[members] @ centroid).tolist()
        # get topk by sim
        top_pairs = sorted(list(zip(members, sims)), key=lambda x: x[1], reverse=True)[:topk]
        exemplars = []
        for mid, sim in top_pairs:
            ent = entities[mid]
            exemplars.append({
                "entity_index": int(mid),
                "entity_id": ent.get("id"),
                "entity_name": ent.get("entity_name"),
                "normalized": normalize_name(ent.get("entity_name") or ""),
                "sim_to_centroid": float(sim)
            })
        out.append({"cluster_id": int(lab), "size": int(size), "exemplars": exemplars})
    return out

# ---------------- Main ----------------
def main():
    print("Loading entities...")
    entities = load_entities(ENTITIES_PATH)
    print(f"Loaded {len(entities)} entities.")

    print("Computing lexical & type stats...")
    lex = basic_lexical_stats(entities)
    types = type_confidence_stats(entities)
    per_chunk, by_chunk = per_chunk_distribution(entities)

    # save CSVs: name frequencies, entities by chunk
    name_rows = []
    for name, cnt in Counter([ (e.get("entity_name") or "").strip() for e in entities if (e.get("entity_name") or "").strip() ]).most_common():
        name_rows.append({"entity_name": name, "count": cnt, "normalized": normalize_name(name)})
    pd.DataFrame(name_rows).to_csv(OUT_DIR / "name_frequencies.csv", index=False)

    chunk_rows = []
    for chunk, lst in by_chunk.items():
        for e in lst:
            chunk_rows.append({
                "chunk_id": chunk,
                "entity_id": e.get("id"),
                "entity_name": e.get("entity_name"),
                "entity_type_hint": e.get("entity_type_hint"),
                "confidence_score": e.get("confidence_score")
            })
    pd.DataFrame(chunk_rows).to_csv(OUT_DIR / "entities_by_chunk.csv", index=False)

    report = {
        "lexical": lex,
        "types": types,
        "per_chunk": per_chunk,
        "n_entities": len(entities)
    }

    # Embedding analysis if present
    print("Attempting to load embeddings...")
    vecs, meta = load_embeddings(EMBEDDINGS_NPY, EMBED_META_NDX if EMBED_META_NDX.exists() else None)
    if vecs is None:
        print("Embeddings not found at:", EMBEDDINGS_NPY)
        report["embeddings_present"] = False
    else:
        print("Embeddings loaded. shape=", vecs.shape)
        report["embeddings_present"] = True
        # normalize & build faiss index
        if faiss is None:
            print("faiss not available; skipping FAISS neighbor stats.")
            report["faiss_available"] = False
        else:
            print("Building FAISS index...")
            idx = build_faiss_index(vecs)
            print("Computing neighbor similarity stats (this may take a while)...")
            emb_stats = embedding_neighbor_stats(vecs, idx, topN=TOPN_NEIGHBORS)
            report["embedding_neighbor_stats"] = emb_stats
            # suggest similarity threshold heuristically:
            avg_top1 = emb_stats.get("avg_top1_similarity", 0.0)
            report["suggested_sim_threshold"] = float(max(0.70, min(0.90, avg_top1 - 0.03)))
            print("avg_top1_similarity:", emb_stats.get("avg_top1_similarity"))

        # clustering
        if hdbscan is None:
            print("hdbscan not available; skipping clustering.")
            report["hdbscan_available"] = False
        else:
            print("Running HDBSCAN clustering (min_cluster_size=%d)..." % MIN_CLUSTER_SIZE)
            clust_res = run_hdbscan(vecs, min_cluster_size=MIN_CLUSTER_SIZE)
            report["hdbscan_result_summary"] = clust_res.get("cluster_counts") if clust_res else None
            if isinstance(clust_res, dict) and "labels" in clust_res:
                labels = clust_res["labels"]
                # save labels mapping (index -> label)
                with open(OUT_DIR / "hdbscan_labels.json", "w", encoding="utf-8") as fh:
                    json.dump({"labels": labels}, fh, indent=2)
                # produce cluster exemplars
                exemplars = cluster_exemplars(labels, entities, vecs, topk=EXEMPLAR_COUNT)
                # save cluster exemplars and top cluster sizes
                with open(OUT_DIR / "hdbscan_clusters_examples.json", "w", encoding="utf-8") as fh:
                    json.dump(exemplars, fh, indent=2, ensure_ascii=False)
                # small CSV of top clusters
                rows = []
                for c in exemplars:
                    rows.append({
                        "cluster_id": c["cluster_id"],
                        "size": c["size"],
                        "exemplar_0": c["exemplars"][0]["entity_name"] if c["exemplars"] else None,
                        "exemplar_1": c["exemplars"][1]["entity_name"] if len(c["exemplars"])>1 else None
                    })
                pd.DataFrame(rows).to_csv(OUT_DIR / "top_clusters_examples.csv", index=False)
                # summary numbers
                num_clusters = len(exemplars)
                largest_clusters = sorted(exemplars, key=lambda x: x["size"], reverse=True)[:20]
                report["hdbscan_clusters_top20"] = [{ "cluster_id":c["cluster_id"], "size":c["size"] } for c in largest_clusters]
                report["hdbscan_n_clusters"] = int(num_clusters)
                # detect large clusters
                report["hdbscan_large_clusters"] = [ {"cluster_id":c["cluster_id"], "size":c["size"]} for c in exemplars if c["size"] >= LARGE_CLUSTER_THRESH ]
            else:
                report["hdbscan_error"] = clust_res

    # write report JSON
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # PRINT short summary
    print("\n=== SUMMARY ===")
    print(f"Entities: {report['n_entities']}")
    print("Unique normalized names (sample top 10):")
    for name,count in report["lexical"]["top_normalized_names"][:10]:
        print(f"  {name}  ({count})")
    print("Top entity types (sample 10):")
    for t,c in report["types"]["type_distribution"][:10]:
        print(f"  {t}: {c}")
    print("Chunks with entities:", report["per_chunk"]["n_chunks_with_entities"])
    if report.get("embeddings_present"):
        if report.get("embedding_neighbor_stats"):
            print("Embedding avg top1 sim:", report["embedding_neighbor_stats"]["avg_top1_similarity"])
            print("Counts above thresholds:", report["embedding_neighbor_stats"]["counts_above"])
            print("Suggested sim threshold (heuristic):", report["suggested_sim_threshold"])
        if report.get("hdbscan_n_clusters"):
            print("HDBSCAN clusters (top):", report["hdbscan_clusters_top20"][:5])
            print("Large clusters (>= %d): %s" % (LARGE_CLUSTER_THRESH, report.get("hdbscan_large_clusters", [])[:5]))

    print("\nOutputs written to:", OUT_DIR)
    print("Main report JSON:", REPORT_JSON)

if __name__ == "__main__":
    main()


#endregion#? 
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   

#!/usr/bin/env python3
"""
entity_embed_neighbor_stats.py

1) Loads provisional entities NDJSON (entities_raw.jsonl)
2) Produces embeddings for each entity (name + short description/context)
3) Saves embeddings and metadata
4) Builds FAISS (if available), queries neighbors, and computes counts above similarity thresholds
5) Writes a JSON report + CSV summary for inspection

Usage:
    python entity_embed_neighbor_stats.py

Outputs (OUT_DIR):
 - entities_emb.npy              (embeddings; row i -> entity i)
 - entities_emb_meta.jsonl       (meta rows matching entities)
 - neighbor_stats.json           (neighbor density diagnostics)
 - neighbor_summary.csv          (per-entity top1/topk counts)
"""

import json, os, sys, math
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# transformers + torch
import torch
from transformers import AutoTokenizer, AutoModel

# optional
try:
    import faiss
except Exception:
    faiss = None

# -------- CONFIG --------
ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL_SMALL = "BAAI/bge-small-en-v1.5"
EMB_MODEL_LARGE = "BAAI/bge-large-en-v1.5"

USE_SMALL_MODEL = True        # switch to False to use large model (more accurate, more RAM)
BATCH_SIZE = 64               # tune for GPU/CPU memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_EMBED_NPY = OUT_DIR / "entities_emb.npy"
SAVE_META = OUT_DIR / "entities_emb_meta.jsonl"
NEIGHBOR_JSON = OUT_DIR / "neighbor_stats.json"
NEIGHBOR_CSV = OUT_DIR / "neighbor_summary.csv"

# FAISS / neighbor params
TOPN = 128
SIM_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]  # thresholds to compute counts for
MIN_REQUIRED_ENTITIES_FOR_FAISS = 2

# -------- helpers --------
def load_entities(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ents.append(json.loads(ln))
            except Exception:
                try:
                    ents.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print("WARN: failed to parse entity line; skipping.")
    return ents

def build_input_text_for_entity(e: Dict) -> str:
    # Compose a compact embedding input using canonical short fields
    name = e.get("entity_name") or e.get("name") or ""
    desc = e.get("entity_description") or e.get("entity_description_short") or e.get("context_phrase") or ""
    # include type hint lightly
    typ = e.get("entity_type_hint") or e.get("type") or ""
    parts = []
    if name:
        parts.append(name)
    if typ:
        parts.append(f"[type:{typ}]")
    if desc:
        parts.append(desc)
    txt = " \n ".join(parts).strip()
    return txt if txt else name or desc or e.get("id","")

@torch.no_grad()
def embed_texts(model_name: str, texts: List[str], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        # pooling: mean over tokens with attention mask
        if hasattr(out, "last_hidden_state"):
            token_emb = out.last_hidden_state  # (B, T, D)
        elif hasattr(out, "hidden_states"):
            token_emb = out.hidden_states[-1]
        else:
            raise RuntimeError("Model output has no last_hidden_state")
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        token_emb = token_emb * mask
        sum_emb = token_emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = sum_emb / denom
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)  # normalize
        all_embs.append(mean_emb.cpu().numpy())
    embs = np.vstack(all_embs).astype("float32")
    return embs

def build_faiss_index(vecs: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss not installed")
    faiss.normalize_L2(vecs)  # ensure unit-norm
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx

# -------- main --------
def main():
    print("Loading entities...")
    entities = load_entities(ENTITIES_PATH)
    n = len(entities)
    print(f"Loaded {n} entities.")

    # prepare input texts in same order
    texts = [build_input_text_for_entity(e) for e in entities]
    # choose model
    model_name = EMB_MODEL_SMALL if USE_SMALL_MODEL else EMB_MODEL_LARGE
    print(f"Using embedding model: {model_name} -> device {DEVICE}. Batching {BATCH_SIZE}.")

    # embed
    embs = embed_texts(model_name, texts, batch_size=BATCH_SIZE, device=DEVICE)
    print("Embeddings shape:", embs.shape)

    # save embeddings and meta (meta preserves id, name, type)
    np.save(str(SAVE_EMBED_NPY), embs)
    print("Saved embeddings to:", SAVE_EMBED_NPY)
    with open(SAVE_META, "w", encoding="utf-8") as fh:
        for e in entities:
            meta = {
                "id": e.get("id"),
                "entity_name": e.get("entity_name"),
                "entity_type_hint": e.get("entity_type_hint"),
                "chunk_id": e.get("chunk_id", None)
            }
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print("Saved embedding meta to:", SAVE_META)

    # neighbor diagnostics with FAISS (if available and >1 entity)
    neighbor_report = {
        "n_entities": n,
        "topn": TOPN,
        "sim_thresholds": SIM_THRESHOLDS,
        "per_entity": []  # will contain dict rows
    }

    if faiss is None or n < MIN_REQUIRED_ENTITIES_FOR_FAISS:
        print("FAISS not available or not enough entities; skipping neighbor stats.")
        neighbor_report["faiss_available"] = False
    else:
        print("Building FAISS index and querying neighbors...")
        idx = build_faiss_index(embs)  # uses normalized vecs internally
        neighbor_report["faiss_available"] = True

        # query in batches
        batch = min(256, n)
        rows = []
        counts_above_global = {str(th): 0 for th in SIM_THRESHOLDS}
        top1_list = []
        for i in tqdm(range(0, n, batch), desc="FAISS query batches"):
            q = embs[i:i+batch]
            D, I = idx.search(q, TOPN+1)  # includes self
            for row_idx_in_batch, (drow, irow) in enumerate(zip(D, I)):
                global_idx = i + row_idx_in_batch
                # drop self (should be near 1.0)
                sims = [float(s) for s in drow if s < 0.9999]
                ids = [int(j) for j in irow[:len(sims)] if j != global_idx]
                top1 = sims[0] if sims else 0.0
                top1_list.append(top1)
                # counts above thresholds
                counts = { str(th): int(sum(1 for s in sims if s >= th)) for th in SIM_THRESHOLDS }
                for th in SIM_THRESHOLDS:
                    if counts[str(th)] > 0:
                        counts_above_global[str(th)] += 1
                rows.append({
                    "entity_index": int(global_idx),
                    "entity_id": entities[global_idx].get("id"),
                    "entity_name": entities[global_idx].get("entity_name"),
                    "top1_sim": float(top1),
                    "neighbors_count_topN": int(len(sims)),
                    **{ f"n_ge_{int(th*100)}": counts[str(th)] for th in SIM_THRESHOLDS }
                })
        # aggregate stats
        top1_arr = np.array(top1_list, dtype="float32") if top1_list else np.array([0.0], dtype="float32")
        neighbor_report["aggregate"] = {
            "avg_top1": float(top1_arr.mean()),
            "median_top1": float(np.median(top1_arr)),
            "p25_top1": float(np.percentile(top1_arr, 25)),
            "p75_top1": float(np.percentile(top1_arr, 75)),
            "counts_entities_with_any_neighbor_above_threshold": counts_above_global
        }
        neighbor_report["per_entity_sample_count"] = len(rows)
        neighbor_report["per_entity_rows_saved"] = str(NEIGHBOR_CSV)
        # save CSV
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(NEIGHBOR_CSV, index=False)
        neighbor_report["per_entity_csv"] = str(NEIGHBOR_CSV)
        neighbor_report["per_entity_rows"] = len(rows)

    # save report
    with open(NEIGHBOR_JSON, "w", encoding="utf-8") as fh:
        json.dump(neighbor_report, fh, indent=2, ensure_ascii=False)

    print("Neighbor report saved to:", NEIGHBOR_JSON)
    if neighbor_report.get("faiss_available"):
        print("Per-entity neighbor CSV:", NEIGHBOR_CSV)
        print("Aggregate top1 avg:", neighbor_report["aggregate"]["avg_top1"])
        print("Counts (entities with >=1 neighbor above thresholds):", neighbor_report["aggregate"]["counts_entities_with_any_neighbor_above_threshold"])
    else:
        print("FAISS not available; only embeddings were computed and saved.")

if __name__ == "__main__":
    main()


#endregion#? 
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   

"""
entity_resolution_cluster_first.py

Cluster-first entity resolution pipeline using embeddings + FAISS and LLM micro-batching.

Outputs:
 - entities_resolved.jsonl (final canonical entities)
 - entities_resolution_history.jsonl (per-entity history)
 - actions_log.jsonl (all LLM-decisions & applied actions)
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict, Counter
import uuid
import faiss
import networkx as nx
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
ENTITIES_PATH = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
EMB_NPY = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb.npy")
OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ENTITIES_OUT = OUT_DIR / "entities_resolved.jsonl"
HISTORY_OUT = OUT_DIR / "entities_resolution_history.jsonl"
ACTIONS_LOG = OUT_DIR / "actions_log.jsonl"

# Tunable thresholds (based on your neighbor stats)
T_BASE = 0.85       # base similarity to link nodes into candidate clusters (recall-friendly)
T_AUTO = 0.92       # high-confidence auto-merge (very tight)
MAX_LLM_BATCH = 12  # maximum cluster size passed to LLM at once
MAX_SUBCLUSTER_SIZE = 40  # if cluster >> this, split by name/embedding
MIN_MERGE_DELTA = 5  # minimum merges to continue another pass
MAX_PASSES = 2

LLM_MODEL = "gpt-4o"
LLM_MAX_TOKENS = 1200
LLM_TEMPERATURE = 0.0

# ---------- UTILITIES ----------
def load_entities(path: Path) -> List[Dict]:
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ents.append(json.loads(line))
            except Exception:
                ents.append(json.loads(line.replace("'", '"')))
    return ents

def normalize_name(s: str) -> str:
    import re
    if s is None:
        return ""
    s2 = s.strip().lower()
    s2 = s2.replace("_", " ").replace("-", " ").replace(".", " ")
    s2 = re.sub(r"\b(v|version)\s*\d+(\.\d+)?\b", "", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def build_faiss_index(vecs: np.ndarray):
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx

# Minimal LLM wrapper — replace with your call_openai implementation if present
def call_llm(prompt: str, model: str = LLM_MODEL, max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> str:
    """
    Replace with your OpenAI client implementation. Must return the raw text (string).
    Expected that the model responds with ONLY a JSON array following the action schema.
    """
    # Example: if you already have client.chat.completions.create wrapper, call it here.
    # For safety in this script, raise error so user connects their own.
    raise RuntimeError("call_llm() not implemented. Please wire this to your OpenAI client (see earlier code).")

# ---------- ACTION EXECUTION ----------
def apply_actions(actions: List[Dict], active_entities: Dict[str, Dict], resolved_entities: Dict[str, Dict], history_out_fh):
    """
    Apply LLM-suggested actions to active_entities, produce resolved_entities, and record history.
    Actions expected to be JSON objects of shape:
      - {"action":"merge_entities", "merged_ids": [...], "canonical_name":"...", "new_description":"..."}
      - {"action":"rename_entity", "entity_id":"En_012", "new_name":"...", "new_description":"..."}
      - {"action":"keep_entity", "entity_id":"En_012"}
    """
    ts = time.time()
    for act in actions:
        a = act.get("action")
        if a == "keep_entity":
            eid = act["entity_id"]
            if eid not in active_entities:
                continue
            e = active_entities.pop(eid)
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            resolved = {
                "id_final": new_id,
                "label": e.get("entity_name"),
                "aliases": [],
                "description": e.get("entity_description") or e.get("context_phrase") or "",
                "source_chunks": [e.get("chunk_id")] if e.get("chunk_id") else [],
                "flag": "resolved_entity",
                "members": [e.get("id")]
            }
            resolved_entities[new_id] = resolved
            # history
            hist = {"ts": ts, "action": "keep_entity", "input_id": eid, "result_id": new_id}
            history_out_fh.write(json.dumps(hist, ensure_ascii=False) + "\n")

        elif a == "rename_entity":
            eid = act["entity_id"]
            if eid not in active_entities:
                continue
            e = active_entities.pop(eid)
            new_label = act.get("new_name", e.get("entity_name"))
            new_desc = act.get("new_description", e.get("entity_description") or e.get("context_phrase",""))
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            resolved = {
                "id_final": new_id,
                "label": new_label,
                "aliases": [e.get("entity_name")] if e.get("entity_name") != new_label else [],
                "description": new_desc,
                "source_chunks": [e.get("chunk_id")] if e.get("chunk_id") else [],
                "flag": "resolved_entity",
                "members": [e.get("id")]
            }
            resolved_entities[new_id] = resolved
            hist = {"ts": ts, "action": "rename_entity", "input_id": eid, "result_id": new_id, "new_name": new_label}
            history_out_fh.write(json.dumps(hist, ensure_ascii=False) + "\n")

        elif a == "merge_entities":
            merge_ids = act.get("merged_ids", [])
            canonical_name = act.get("canonical_name") or act.get("canonical_id") or "merged_entity"
            new_desc = act.get("new_description", "")
            members = [ active_entities.pop(mid) for mid in merge_ids if mid in active_entities ]
            if not members:
                continue
            # combine aliases
            aliases = [m.get("entity_name") for m in members if m.get("entity_name") and m.get("entity_name") != canonical_name]
            source_chunks = sorted(list({ m.get("chunk_id") for m in members if m.get("chunk_id") }))
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            resolved = {
                "id_final": new_id,
                "label": canonical_name,
                "aliases": aliases,
                "description": new_desc,
                "source_chunks": source_chunks,
                "flag": "resolved_entity",
                "members": [m.get("id") for m in members]
            }
            resolved_entities[new_id] = resolved
            hist = {"ts": ts, "action": "merge_entities", "merged_ids": merge_ids, "result_id": new_id, "canonical_name": canonical_name}
            history_out_fh.write(json.dumps(hist, ensure_ascii=False) + "\n")

        else:
            # unknown action: ignore but log
            hist = {"ts": ts, "action": "unknown_action", "payload": act}
            history_out_fh.write(json.dumps(hist, ensure_ascii=False) + "\n")

# ---------- PROMPT / LLM formatting ----------
def build_cluster_prompt(cluster_entities: List[Dict], similar_chunks: List[Dict] = None) -> str:
    """
    Build a prompt for the LLM asking to resolve the small group of entities.
    The LLM must return only a JSON array of actions (merge/rename/keep).
    Keep prompt compact but informative: include entity name, short desc, type hint, sample context.
    """
    header = [
        "You are an Entity Resolution Agent.",
        "Task: Given the following candidate entity mentions (surface forms + short descriptions),",
        "decide which mentions refer to the same real-world entity (merge), which should be renamed to avoid confusion,",
        "and which should be kept separate.",
        "Return ONLY a JSON array of actions (no extra text).",
        "Allowed actions:",
        " - merge_entities: {action:'merge_entities', merged_ids:[...], canonical_name:'...', new_description:'...'}",
        " - rename_entity:  {action:'rename_entity', entity_id:'En_xxx', new_name:'..', new_description:'...'}",
        " - keep_entity:    {action:'keep_entity', entity_id:'En_xxx'}",
        "",
        "Rules:",
        "- Do NOT create classes; operate on mentions/instances only.",
        "- Prefer precision: if ambiguous, keep separate rather than incorrectly merging.",
        "- If multiple mentions are the same entity, merge them providing a canonical_name and a one-sentence new_description.",
        "- Use the provided short descriptions to disambiguate; include only mentions present in the input list.",
        ""
    ]
    lines = header[:]
    lines.append("INPUT ENTITIES:")
    for e in cluster_entities:
        lines.append(json.dumps({
            "id": e.get("id"),
            "entity_name": e.get("entity_name"),
            "type_hint": e.get("entity_type_hint"),
            "short_desc": (e.get("entity_description") or e.get("context_phrase") or "")[:240],
            "chunk_id": e.get("chunk_id")
        }, ensure_ascii=False))
    if similar_chunks:
        lines.append("")
        lines.append("SIMILAR CHUNKS (for provenance context):")
        for c in similar_chunks[:5]:
            lines.append((c.get("text") or "")[:300])

    lines.append("")
    lines.append("Return only JSON array of actions.")
    return "\n".join(lines)

# ---------- CLUSTERING helpers ----------
def build_similarity_graph(vecs: np.ndarray, threshold: float = T_BASE, topk_search: int = 64) -> nx.Graph:
    """
    Build an undirected graph where nodes are indices 0..n-1 and there is an edge (i,j)
    if cosine similarity(i,j) >= threshold. Use FAISS to get topk neighbors and filter by threshold.
    """
    n = vecs.shape[0]
    idx = build_faiss_index(vecs)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    batch = min(256, n)
    for i in range(0, n, batch):
        q = vecs[i:i+batch]
        D, I = idx.search(q, topk_search+1)  # includes self
        for row_idx, (drow, irow) in enumerate(zip(D, I)):
            src = i + row_idx
            for sim, dst in zip(drow, irow):
                if dst == src:
                    continue
                if float(sim) >= threshold:
                    G.add_edge(int(src), int(dst), weight=float(sim))
    return G

def split_large_cluster(indices: List[int], vecs: np.ndarray, max_size: int = MAX_LLM_BATCH) -> List[List[int]]:
    """
    Split a very large cluster into subclusters using KMeans where k = ceil(size / max_size).
    Returns list of index groups (indices into vecs and entities).
    """
    size = len(indices)
    if size <= max_size:
        return [indices]
    k = math.ceil(size / max_size)
    subvecs = vecs[indices]
    # KMeans in original embedding space (vecs assumed normalized; euclidean okay)
    kms = KMeans(n_clusters=k, random_state=0)
    labels = kms.fit_predict(subvecs)
    groups = defaultdict(list)
    for idx_pos, lab in enumerate(labels):
        groups[int(lab)].append(indices[idx_pos])
    return list(groups.values())

# ---------- MAIN RESOLUTION PIPELINE ----------
def cluster_first_resolution(entities: List[Dict], vecs: np.ndarray):
    n = len(entities)
    # Map index -> entity id
    idx2id = [e.get("id") for e in entities]

    # Keep active_entities as dict id->entity (shallow copy)
    active_entities = { e.get("id"): dict(e) for e in entities }
    resolved_entities = {}  # new canonical id -> resolved dict

    # history file
    hist_fh = open(HISTORY_OUT, "w", encoding="utf-8")
    actions_fh = open(ACTIONS_LOG, "w", encoding="utf-8")

    total_merges = 0
    pass_no = 0

    while pass_no < MAX_PASSES:
        pass_no += 1
        print(f"\n=== PASS {pass_no} (active_entities={len(active_entities)}) ===")
        # Build index of currently active entity indices and vectors
        id_to_idx = {}
        idx_list = []
        vec_list = []
        for idx, e in enumerate(entities):
            eid = e.get("id")
            if eid in active_entities:
                id_to_idx[eid] = len(idx_list)
                idx_list.append(eid)
                vec_list.append(vecs[idx])
        if not vec_list:
            break
        cur_vecs = np.vstack(vec_list).astype("float32")
        # Build similarity graph
        print("Building similarity graph (threshold:", T_BASE, ") ...")
        G = build_similarity_graph(cur_vecs, threshold=T_BASE, topk_search=128)
        components = list(nx.connected_components(G))
        print(f"Found {len(components)} candidate clusters")

        merges_this_pass = 0

        # process components
        for comp in components:
            comp = sorted(list(comp))
            if len(comp) == 1:
                # skip singletons (keep them as-is)
                continue

            # map comp indices back to original entity ids & their full records
            comp_entity_ids = [ idx_list[i] for i in comp ]
            comp_entities = [ active_entities[eid] for eid in comp_entity_ids ]

            # quick auto-merge: if cluster size small and internal sim stats very high -> auto-merge
            # compute pairwise similarities from graph edges (weights)
            sims = []
            for i in comp:
                for j in comp:
                    if i < j and G.has_edge(i,j):
                        sims.append(G[i][j]['weight'])
            median_sim = float(np.median(sims)) if sims else 0.0

            if median_sim >= T_AUTO and len(comp_entities) <= 8:
                # auto-merge into canonical name = most common normalized name or longest label
                names = [normalize_name(e.get("entity_name") or "") for e in comp_entities]
                common = Counter(names).most_common(1)
                canonical = common[0][0] if common else normalize_name(comp_entities[0].get("entity_name") or "")
                canonical_label = canonical
                new_desc = "Auto-merged cluster (high similarity)."
                action = {
                    "action": "merge_entities",
                    "merged_ids": [e.get("id") for e in comp_entities],
                    "canonical_name": canonical_label,
                    "new_description": new_desc
                }
                # apply action
                try:
                    apply_actions([action], active_entities, resolved_entities, hist_fh)
                    actions_fh.write(json.dumps({"pass":pass_no, "auto_action": action}, ensure_ascii=False) + "\n")
                    merges_this_pass += 1
                except Exception as ex:
                    print("Auto-merge failed:", ex)
                continue

            # Otherwise resolve with LLM. If cluster is large, split
            if len(comp_entities) > MAX_LLM_BATCH:
                # split into subclusters by KMeans on the comp indices
                sub_groups = split_large_cluster(comp, cur_vecs, max_size=MAX_LLM_BATCH)
            else:
                sub_groups = [comp]

            for sg in sub_groups:
                sg_entity_ids = [ idx_list[i] for i in sg ]
                sg_entities = [ active_entities[eid] for eid in sg_entity_ids ]
                # build prompt
                prompt = build_cluster_prompt(sg_entities)
                # call LLM
                try:
                    raw = call_llm(prompt, model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
                except Exception as e:
                    print("LLM call failed (user must wire the call):", e)
                    # fallback: skip this subgroup
                    continue

                # parse JSON
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        raise ValueError("LLM did not return a JSON list")
                except Exception as e:
                    # log raw output and skip
                    actions_fh.write(json.dumps({"pass":pass_no, "prompt": prompt[:800], "raw_output": raw[:2000], "error": str(e)}, ensure_ascii=False) + "\n")
                    continue

                # apply actions
                apply_actions(parsed, active_entities, resolved_entities, hist_fh)
                actions_fh.write(json.dumps({"pass":pass_no, "llm_actions": parsed}, ensure_ascii=False) + "\n")
                # count merges
                merges_here = sum(1 for a in parsed if a.get("action") == "merge_entities")
                merges_this_pass += merges_here

        print(f"Pass {pass_no} merges_this_pass = {merges_this_pass}")
        total_merges += merges_this_pass

        # stop condition
        if merges_this_pass < MIN_MERGE_DELTA:
            print("Merges below MIN_MERGE_DELTA; stopping further passes.")
            break
        if pass_no >= MAX_PASSES:
            break

    # cleanup: any remaining active_entities -> keep them as resolved singletons
    with open(HISTORY_OUT, "a", encoding="utf-8") as hist_fh_final:
        for eid, e in list(active_entities.items()):
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            resolved_entities[new_id] = {
                "id_final": new_id,
                "label": e.get("entity_name"),
                "aliases": [],
                "description": e.get("entity_description") or e.get("context_phrase") or "",
                "source_chunks": [e.get("chunk_id")] if e.get("chunk_id") else [],
                "flag": "resolved_entity",
                "members": [e.get("id")]
            }
            hist = {"ts": time.time(), "action": "auto_keep_singleton", "input_id": eid, "result_id": new_id}
            hist_fh_final.write(json.dumps(hist, ensure_ascii=False) + "\n")

    # write final resolved entities file
    with open(FINAL_ENTITIES_OUT, "w", encoding="utf-8") as fh:
        for rid, r in resolved_entities.items():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    hist_fh.close()
    actions_fh.close()
    print("Resolution finished. total_resolved:", len(resolved_entities))
    print("Final entities saved to:", FINAL_ENTITIES_OUT)
    print("History saved to:", HISTORY_OUT)
    print("Actions log saved to:", ACTIONS_LOG)

# ---------- RUN ----------
if __name__ == "__main__":
    # Load
    print("Loading entities and embeddings...")
    entities = load_entities(ENTITIES_PATH)
    vecs = np.load(str(EMB_NPY)).astype("float32")  # rows must align with entities list
    if vecs.shape[0] != len(entities):
        raise RuntimeError("Embedding rows do not match entities count! rows=%d entities=%d" % (vecs.shape[0], len(entities)))
    # run pipeline
    cluster_first_resolution(entities, vecs)


#endregion#? 
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   

#!/usr/bin/env python3
"""
inspect_merge_actions.py

Inspect actions_log.jsonl for merge_entities actions, summarize, and optionally
compute embedding-based sanity checks for each merge.

Outputs:
 - merge_summary.csv   (one row per merge action)
 - merge_summary.json  (detailed JSON)
"""

import json
from pathlib import Path
from collections import Counter
import itertools
import csv
import numpy as np

# ---------- CONFIG -> adjust if your files are elsewhere ----------
ACTIONS_LOG = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/actions_log.jsonl")
ENTITIES_FILE = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
EMB_NPY = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/clusterring/entities_emb.npy")  # may be absent
OUT_DIR = ACTIONS_LOG.parent
MERGE_SUM_CSV = OUT_DIR / "merge_summary.csv"
MERGE_SUM_JSON = OUT_DIR / "merge_summary.json"

# ---------- Helpers ----------
def load_actions(path):
    acts = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                acts.append(json.loads(ln))
            except Exception:
                # tolerant parse
                try:
                    acts.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print("Warning: failed to parse line in actions log.")
    return acts

def extract_merge_actions(actions):
    merges = []
    for a in actions:
        # The log may contain different envelope keys; handle both direct and nested
        if a.get("auto_action") and isinstance(a["auto_action"], dict):
            a2 = a["auto_action"]
        elif a.get("llm_actions") and isinstance(a["llm_actions"], list):
            # sometimes we logged a group; need to iterate
            for sub in a["llm_actions"]:
                if sub.get("action") == "merge_entities":
                    merges.append(sub)
            continue
        else:
            a2 = a

        if a2.get("action") == "merge_entities":
            merges.append(a2)
    return merges

def load_entities_list(path):
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ents.append(json.loads(ln))
            except Exception:
                ents.append(json.loads(ln.replace("'", '"')))
    return ents

def build_entity_index_map(entities):
    # Maps entity_id -> row_index (0-based)
    m = {}
    for i, e in enumerate(entities):
        eid = e.get("id")
        if eid in m:
            # if duplicate ids (shouldn't happen) keep first
            continue
        m[eid] = i
    return m

def mean_pairwise_cosine(vecs):
    # expects normalized vectors (or will normalize inside)
    if vecs.shape[0] <= 1:
        return 1.0
    # normalize
    v = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim_mat = v @ v.T
    # take upper triangular off-diagonal
    n = vecs.shape[0]
    idxs = np.triu_indices(n, k=1)
    vals = sim_mat[idxs]
    return float(vals.mean()) if vals.size > 0 else 1.0

# ---------- MAIN ----------
def main():
    if not ACTIONS_LOG.exists():
        print("Actions log not found:", ACTIONS_LOG)
        return

    actions = load_actions(ACTIONS_LOG)
    merges = extract_merge_actions(actions)
    print(f"Total actions lines loaded: {len(actions)}. Merge actions extracted: {len(merges)}")

    # Basic merge stats
    sizes = [ len(m.get("merged_ids", [])) for m in merges ]
    size_counter = Counter(sizes)
    print("Merge size distribution (size:count):")
    for s,c in sorted(size_counter.items()):
        print(f"  {s}: {c}")

    # Load entities and embeddings if possible
    entities = []
    id_to_idx = {}
    emb_matrix = None
    if ENTITIES_FILE.exists():
        entities = load_entities_list(ENTITIES_FILE)
        id_to_idx = build_entity_index_map(entities)
        print(f"Loaded {len(entities)} entities from {ENTITIES_FILE}")
    else:
        print("Entities file not found at:", ENTITIES_FILE)

    if EMB_NPY.exists():
        emb_matrix = np.load(str(EMB_NPY)).astype("float32")
        print("Embeddings loaded shape:", emb_matrix.shape)
        if emb_matrix.shape[0] != len(entities):
            print("WARNING: embeddings row count != entities count. Embedding-based checks may be invalid.")
    else:
        print("Embeddings not found at:", EMB_NPY, "-- skipping similarity checks.")

    # Prepare per-merge rows
    rows = []
    suspicious = []
    for i, m in enumerate(merges):
        merged_ids = m.get("merged_ids") or m.get("merged", []) or []
        canonical = m.get("canonical_name") or m.get("canonical", "") or m.get("canonical_id", "")
        result_id = m.get("result_id") or m.get("id") or ""
        size = len(merged_ids)
        # get member names if entities loaded
        member_names = []
        mean_sim = None
        missing_indexes = []
        for mid in merged_ids:
            if id_to_idx and mid in id_to_idx:
                idx = id_to_idx[mid]
                member_names.append(entities[idx].get("entity_name"))
            else:
                member_names.append(None)
                missing_indexes.append(mid)
        # similarity check
        if emb_matrix is not None and id_to_idx:
            valid_idxs = [id_to_idx[mid] for mid in merged_ids if mid in id_to_idx]
            if len(valid_idxs) >= 1:
                vecs = emb_matrix[valid_idxs]
                mean_sim = mean_pairwise_cosine(vecs) if len(valid_idxs) > 1 else 1.0
            else:
                mean_sim = None

        row = {
            "merge_action_index": i,
            "result_id": result_id,
            "canonical_name": canonical,
            "merged_ids": merged_ids,
            "merged_names": member_names,
            "size": size,
            "mean_pairwise_sim": mean_sim,
            "missing_entity_ids": missing_indexes
        }
        rows.append(row)
        # flag suspicious merges: size>6 or mean_sim exists and <0.75
        if (size > 6) or (mean_sim is not None and mean_sim < 0.75):
            suspicious.append(row)

    # write CSV
    csv_fields = ["merge_action_index","result_id","canonical_name","size","mean_pairwise_sim","merged_ids","merged_names","missing_entity_ids"]
    with open(MERGE_SUM_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            # flatten lists to string
            r2 = dict(r)
            r2["merged_ids"] = ";".join(r2["merged_ids"]) if r2["merged_ids"] else ""
            r2["merged_names"] = ";".join([n if n else "NULL" for n in r2["merged_names"]]) if r2["merged_names"] else ""
            r2["missing_entity_ids"] = ";".join(r2["missing_entity_ids"]) if r2["missing_entity_ids"] else ""
            writer.writerow(r2)

    # write JSON for deeper inspection
    with open(MERGE_SUM_JSON, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    print(f"Merge summary written to CSV: {MERGE_SUM_CSV} and JSON: {MERGE_SUM_JSON}")
    print(f"Total merge actions: {len(rows)}")
    print("Top 10 largest merges (by size):")
    for r in sorted(rows, key=lambda x: x["size"], reverse=True)[:10]:
        print(f"  size={r['size']}, canonical='{r['canonical_name']}', ids={r['merged_ids'][:8]}... mean_sim={r['mean_pairwise_sim']}")

    if suspicious:
        print("\nSuspicious merges detected (size>6 or mean_sim<0.75):")
        for s in suspicious[:20]:
            print(f"  size={s['size']}, canonical='{s['canonical_name']}', mean_sim={s['mean_pairwise_sim']}, missing={s['missing_entity_ids']}")

    # simple aggregates
    total_merged_ids = sum(r["size"] for r in rows)
    unique_merged_member_ids = set(itertools.chain.from_iterable(r["merged_ids"] for r in rows))
    print(f"\nAggregate stats: total merged mentions across actions = {total_merged_ids}")
    print(f"Unique merged member ids (count) = {len(unique_merged_member_ids)}")
    print("Done.")

if __name__ == "__main__":
    main()


#endregion#? 
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   



#endregion#? 
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   



#endregion#? 
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   



#endregion#? 
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
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # change to large if you want and have resources
USE_CUDA = True  # set False to force CPU
BATCH_EMBED = 64

# Neighbor retrieval settings
TOPN_FAST = 64   # initial retrieval size (fast)
T_BASE = 0.85    # base graph threshold for forming clusters (recall-friendly)
T_AUTO = 0.92    # auto-merge threshold (very safe)
T_HIGH = 0.95    # very high similarity (definitely same)
CROSS_ENCODER_RANGE = (0.80, 0.92)  # range where we prefer LLM rerankers (we will use cluster-LLM instead)

# Composite similarity weights (name, desc, ctx, type)
WEIGHTS = {"name": 0.45, "desc": 0.25, "ctx": 0.2, "type": 0.1}

# LLM settings
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be present to use LLM decisions
MAX_LLM_BATCH = 10  # maximum cluster size to feed LLM intact; larger clusters are split
MAX_PASSES = 2
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
#region:#?   

#!/usr/bin/env python3
"""
analyze_entity_resolution_detailed.py

Detailed analysis + audit artifacts after entity resolution.

Inputs (expected):
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/entities_resolved.jsonl
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/resolve_map.json
  - /home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/clusters_for_review.jsonl (optional)

Outputs (all under OUT_DIR):
  - merged_members.csv            (rows = original mentions that have a resolved_id)
  - all_members_with_resolution.csv  (rows = all original mentions with resolution data)
  - clusters_detailed.jsonl       (one JSON object per resolved canonical node with full member records)
  - clusters_summary.csv          (one row per resolved canonical node with aggregated stats)
  - clusters_flagged_for_review.csv (if any)
  - singletons_unresolved.csv     (original mentions not assigned to any resolved node)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import statistics

# --- CONFIG: adjust paths if needed ---
ENT_RAW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG/entities_raw.jsonl")
RESOLVED_JSONL = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/entities_resolved.jsonl")
RESOLVE_MAP = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/resolve_map.json")
CLUSTERS_REVIEW = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/clusters_for_review.jsonl")

OUT_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/entity-resolved/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- helper loaders ---
def load_jsonl(path):
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
                # tolerant fallback
                try:
                    items.append(json.loads(ln.replace("'", '"')))
                except Exception:
                    print(f"[warn] failed to parse line in {path}: {ln[:120]}")
    return items

def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def safe_get(e, key, default=""):
    v = e.get(key)
    if v is None:
        return default
    return v

def normalize_surface(s):
    if not s:
        return ""
    return " ".join(str(s).strip().lower().split())

# --- load data ---
print("Loading original mentions...")
entities_raw = load_jsonl(ENT_RAW)
print(f" - original mentions: {len(entities_raw)}")

print("Loading resolved canonical nodes...")
resolved = load_jsonl(RESOLVED_JSONL)
print(f" - resolved canonical nodes: {len(resolved)}")

resolve_map = load_json(RESOLVE_MAP) or {}
print(f" - resolve_map entries: {len(resolve_map)}")

clusters_review = load_jsonl(CLUSTERS_REVIEW)
print(f" - clusters_for_review entries: {len(clusters_review)}")

# build lookups
entities_by_id = { e.get("id"): e for e in entities_raw }
resolved_by_id = { r.get("id_final"): r for r in resolved }

# build inverted mapping resolved_id -> member ids (from resolve_map)
members_by_resolved = defaultdict(list)
unmapped_originals = []
for orig_id in entities_by_id.keys():
    resolved_id = resolve_map.get(orig_id)
    if resolved_id:
        members_by_resolved[resolved_id].append(orig_id)
    else:
        unmapped_originals.append(orig_id)

# Safety: if resolved canonical records contain a members list already, prefer that
# (ensures consistency if resolve_map incomplete)
for r in resolved:
    rid = r.get("id_final")
    if rid:
        m = r.get("members")
        if m and isinstance(m, list):
            # merge uniqueness
            for mid in m:
                if mid not in members_by_resolved[rid]:
                    members_by_resolved[rid].append(mid)

# --- Produce merged_members.csv (ONLY merged members, i.e., those with a resolved id) ---
merged_csv = OUT_DIR / "merged_members.csv"
all_members_csv = OUT_DIR / "all_members_with_resolution.csv"
clusters_jsonl = OUT_DIR / "clusters_detailed.jsonl"
clusters_summary_csv = OUT_DIR / "clusters_summary.csv"
singletons_csv = OUT_DIR / "singletons_unresolved.csv"
flagged_review_csv = OUT_DIR / "clusters_flagged_for_review.csv"

# Write merged_members.csv and all_members_with_resolution.csv
fieldnames = [
    "original_id",
    "entity_name",
    "entity_description",
    "context_phrase",
    "used_context_excerpt",
    "entity_type_hint",
    "chunk_id",
    "confidence_score",
    "resolved_id",
    "resolved_label",
    "resolved_size",
    "resolved_aliases"
]

with open(merged_csv, "w", encoding="utf-8", newline="") as fh_m, \
     open(all_members_csv, "w", encoding="utf-8", newline="") as fh_all:
    writer_m = csv.DictWriter(fh_m, fieldnames=fieldnames)
    writer_all = csv.DictWriter(fh_all, fieldnames=fieldnames)
    writer_m.writeheader()
    writer_all.writeheader()

    # for convenience, build resolved label/size/aliases lookup
    resolved_info = {}
    for rid, members in members_by_resolved.items():
        rec = resolved_by_id.get(rid, {})
        resolved_info[rid] = {
            "label": rec.get("label") or rid,
            "size": len(members),
            "aliases": rec.get("aliases") or []
        }

    for orig in entities_raw:
        oid = orig.get("id")
        rid = resolve_map.get(oid)
        rlabel = ""
        rsize = ""
        raliases = ""
        if rid and rid in resolved_info:
            rlabel = resolved_info[rid]["label"]
            rsize = resolved_info[rid]["size"]
            raliases = "|".join(resolved_info[rid]["aliases"]) if resolved_info[rid]["aliases"] else ""
        row = {
            "original_id": oid,
            "entity_name": safe_get(orig, "entity_name", ""),
            "entity_description": safe_get(orig, "entity_description", ""),
            "context_phrase": safe_get(orig, "context_phrase", ""),
            "used_context_excerpt": safe_get(orig, "used_context_excerpt", ""),
            "entity_type_hint": safe_get(orig, "entity_type_hint", ""),
            "chunk_id": safe_get(orig, "chunk_id", ""),
            "confidence_score": safe_get(orig, "confidence_score", ""),
            "resolved_id": rid or "",
            "resolved_label": rlabel,
            "resolved_size": rsize,
            "resolved_aliases": raliases
        }
        # write to all-members table
        writer_all.writerow(row)
        # if resolved, also write to merged_members
        if rid:
            writer_m.writerow(row)

print("Wrote merged members CSV:", merged_csv)
print("Wrote all members CSV:", all_members_csv)

# --- clusters_detailed.jsonl (one canonical node per line with full member records) ---
with open(clusters_jsonl, "w", encoding="utf-8") as fh:
    for rid, members in members_by_resolved.items():
        rec = resolved_by_id.get(rid, {})
        cluster_obj = {
            "resolved_id": rid,
            "label": rec.get("label") or rid,
            "description": rec.get("description") or "",
            "aliases": rec.get("aliases") or [],
            "members_count": len(members),
            "members": []
        }
        for mid in members:
            orig = entities_by_id.get(mid, {})
            # include all original fields for inspection
            member_record = {
                "original_id": mid,
                "entity_name": orig.get("entity_name",""),
                "entity_description": orig.get("entity_description",""),
                "context_phrase": orig.get("context_phrase",""),
                "used_context_excerpt": orig.get("used_context_excerpt",""),
                "entity_type_hint": orig.get("entity_type_hint",""),
                "chunk_id": orig.get("chunk_id",""),
                "confidence_score": orig.get("confidence_score",""),
                "_raw": orig.get("_raw_llm") or None
            }
            cluster_obj["members"].append(member_record)
        fh.write(json.dumps(cluster_obj, ensure_ascii=False) + "\n")

print("Wrote clusters detailed JSONL:", clusters_jsonl)

# --- clusters_summary.csv (one row per canonical node) ---
summary_fields = [
    "resolved_id",
    "label",
    "size",
    "mean_confidence",
    "median_confidence",
    "most_common_surface",
    "most_common_surface_count",
    "distinct_types_count",
    "top_types",
    "aliases"
]
with open(clusters_summary_csv, "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=summary_fields)
    writer.writeheader()
    for rid, members in members_by_resolved.items():
        rec = resolved_by_id.get(rid, {})
        surfaces = []
        confidences = []
        types = []
        for mid in members:
            m = entities_by_id.get(mid, {})
            surfaces.append(safe_get(m, "entity_name", ""))
            cs = m.get("confidence_score")
            try:
                if cs is not None and cs != "":
                    confidences.append(float(cs))
            except Exception:
                pass
            t = safe_get(m, "entity_type_hint", "")
            if t:
                types.append(t)
        most_common_surface = ""
        most_common_count = 0
        if surfaces:
            normed = [normalize_surface(s) for s in surfaces if s]
            if normed:
                mc = Counter(normed).most_common(1)[0]
                most_common_surface, most_common_count = mc[0], mc[1]
        mean_conf = statistics.mean(confidences) if confidences else ""
        median_conf = statistics.median(confidences) if confidences else ""
        type_counts = Counter(types)
        top_types = ";".join([f"{t}:{c}" for t, c in type_counts.most_common(5)])
        writer.writerow({
            "resolved_id": rid,
            "label": rec.get("label") or rid,
            "size": len(members),
            "mean_confidence": mean_conf,
            "median_confidence": median_conf,
            "most_common_surface": most_common_surface,
            "most_common_surface_count": most_common_count,
            "distinct_types_count": len(type_counts),
            "top_types": top_types,
            "aliases": "|".join(rec.get("aliases") or [])
        })

print("Wrote clusters summary CSV:", clusters_summary_csv)

# --- clusters_flagged_for_review.csv (if present) ---
if clusters_review:
    # clusters_review lines might be dicts with 'cluster' and 'mean_sim' or reason; normalize
    with open(flagged_review_csv, "w", encoding="utf-8", newline="") as fh:
        fields = ["cluster_id", "size", "mean_sim", "reason", "sample_members"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(clusters_review):
            cluster = item.get("cluster") or item.get("members") or item
            if isinstance(cluster, dict):
                # sometimes already has metadata
                members = cluster.get("members") or cluster.get("cluster") or []
            elif isinstance(cluster, list):
                members = cluster
            else:
                members = []
            mean_sim = item.get("mean_sim") or item.get("avg_sim") or ""
            reason = item.get("reason") or item.get("flag") or ""
            sample_members = ";".join(cluster[:6]) if cluster else ""
            writer.writerow({
                "cluster_id": f"flag_{i}",
                "size": len(members),
                "mean_sim": mean_sim,
                "reason": reason,
                "sample_members": sample_members
            })
    print("Wrote flagged-for-review CSV:", flagged_review_csv)
else:
    print("No clusters_for_review present; skipped writing flagged review CSV.")

# --- singletons_unresolved.csv (original mentions not mapped to any resolved node) ---
with open(singletons_csv, "w", encoding="utf-8", newline="") as fh:
    fields2 = ["original_id", "entity_name", "entity_description", "entity_type_hint", "chunk_id", "confidence_score"]
    writer = csv.DictWriter(fh, fieldnames=fields2)
    writer.writeheader()
    for oid in unmapped_originals:
        o = entities_by_id.get(oid, {})
        writer.writerow({
            "original_id": oid,
            "entity_name": safe_get(o, "entity_name", ""),
            "entity_description": safe_get(o, "entity_description", ""),
            "entity_type_hint": safe_get(o, "entity_type_hint", ""),
            "chunk_id": safe_get(o, "chunk_id", ""),
            "confidence_score": safe_get(o, "confidence_score", "")
        })
print("Wrote singletons_unresolved.csv:", singletons_csv)

print("\nDONE. Analysis outputs in:", OUT_DIR)
print("Files produced:")
for p in [merged_csv, all_members_csv, clusters_jsonl, clusters_summary_csv, singletons_csv]:
    print(" -", p)
if clusters_review:
    print(" -", flagged_review_csv)


#endregion#? 
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Entity Resolution V0

#endregion#? Entity Resolution V0
#?#########################  End  ##########################


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











