
# # !pip install pymupdf openai

# #load openai key using dotenv
# from dotenv import load_dotenv
# import os
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY




# #!/usr/bin/env python3
# """
# pdf_multimodal_extras.py

# Purpose:
#   Render PDF pages to images and use a vision-capable OpenAI model (gpt-4.1 / gpt-5.1)
#   to extract ONLY non-plain-text information present visually:
#     - tables (markdown + row-wise sentences)
#     - figures/diagrams (caption + description + key entities/relations)
#     - equations/formulas (LaTeX + plain-language reading + variables)
#     - charts/plots (summary / extracted series if clear)
#     - truth tables / logical tables

# Outputs:
#   - <out_prefix>_extras.txt      human-readable per-page results
#   - <out_prefix>_extras.jsonl    raw per-page records (JSONL)
#   - optional: merges extras into a Plain_Text.json if --merge_plain_text is set

# Requirements:
#   pip install pymupdf openai
#   export OPENAI_API_KEY="sk-..."
#   (Use GPT-4.1 or GPT-5.1 models that support image inputs.)
# """

# import argparse
# import base64
# import json
# import os
# import sys
# import time
# from pathlib import Path
# from typing import Optional

# import fitz  # PyMuPDF
# from openai import OpenAI

# # ---- System instructions for the vision model ----
# SYSTEM_INSTRUCTIONS = (
#     "You are a precise document-vision assistant. "
#     "You will receive 1) MACHINE_TEXT (plain text extracted programmatically from the page) "
#     "and 2) PAGE_IMAGE (the page rendered as an image).\n\n"
#     "TASK: Output ONLY non-plain-text content that is present visually but NOT captured in MACHINE_TEXT. "
#     "Focus on: TABLES, FIGURES/DIAGRAMS, EQUATIONS/FORMULAS, CHARTS/PLOTS, and TRUTH_TABLES.\n\n"
#     "RULES:\n"
#     "- If nothing missing, output exactly: NO_NON_TEXT_ELEMENTS\n"
#     "- Use tags to start each block: [TABLE], [FIGURE], [EQUATION], [CHART], [DIAGRAM], [TRUTH_TABLE]\n"
#     "- For [TABLE]: include a compact markdown table (if feasible) and 1-8 row-wise sentence facts.\n"
#     "- For [EQUATION]: provide LaTeX if possible and a short plain-language reading + variable list.\n"
#     "- For [FIGURE]/[DIAGRAM]: include caption (if present) then 1-3 concise sentences describing key entities/relations.\n"
#     "- Keep outputs short, sentence-friendly, and page-by-page.\n"
#     "- Do NOT re-output normal body paragraphs already present in MACHINE_TEXT.\n"
# )


# def png_bytes_to_data_url(png_bytes: bytes) -> str:
#     return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# def render_page_to_png_bytes(page: fitz.Page, dpi: int = 200) -> bytes:
#     pix = page.get_pixmap(dpi=dpi, alpha=False)
#     return pix.tobytes("png")


# def extract_machine_text(page: fitz.Page, max_chars: int = 12000) -> str:
#     txt = page.get_text("text") or ""
#     txt = txt.strip()
#     if len(txt) > max_chars:
#         txt = txt[:max_chars] + "\n[TRUNCATED]"
#     return txt


# def parse_response_text(resp) -> str:
#     """
#     Robustly extract textual output from OpenAI Responses object.
#     Uses resp.output_text when available, otherwise assembles strings from resp.output.
#     """
#     try:
#         # New OpenAI client often provides output_text
#         if hasattr(resp, "output_text") and resp.output_text:
#             return resp.output_text.strip()
#         # fallback: resp.output may be list of message/blocks
#         out = ""
#         if hasattr(resp, "output") and resp.output:
#             for item in resp.output:
#                 # item might be dict with 'content' list
#                 if isinstance(item, dict):
#                     content = item.get("content")
#                     if isinstance(content, list):
#                         for c in content:
#                             if isinstance(c, dict) and c.get("type") == "output_text":
#                                 out += c.get("text", "")
#                             elif isinstance(c, str):
#                                 out += c
#                     elif isinstance(content, str):
#                         out += content
#                 elif isinstance(item, str):
#                     out += item
#         if out:
#             return out.strip()
#     except Exception:
#         pass
#     # last fallback: stringify
#     try:
#         return str(resp)
#     except Exception:
#         return ""


# def call_vision_model(
#     client: OpenAI,
#     model: str,
#     machine_text: str,
#     image_data_url: str,
#     max_output_tokens: int = 2000,
#     retries: int = 2,
#     backoff_s: float = 0.5,
# ) -> str:
#     user_prompt = (
#         "MACHINE_TEXT (plain text extracted programmatically from this page):\n"
#         "```\n"
#         f"{machine_text if machine_text else '[EMPTY]'}\n"
#         "```\n\n"
#         "Now inspect the PAGE_IMAGE provided and output ONLY what is missing from MACHINE_TEXT. "
#         "If nothing is missing, write exactly: NO_NON_TEXT_ELEMENTS\n\n"
#         "Remember to use the block tags: [TABLE], [FIGURE], [EQUATION], [CHART], [DIAGRAM], [TRUTH_TABLE]."
#     )

#     attempt = 0
#     last_exc = None
#     while attempt <= retries:
#         try:
#             resp = client.responses.create(
#                 model=model,
#                 instructions=SYSTEM_INSTRUCTIONS,
#                 input=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "input_text", "text": user_prompt},
#                             {"type": "input_image", "image_url": image_data_url},
#                         ],
#                     }
#                 ],
#                 max_output_tokens=max_output_tokens,
#             )
#             txt = parse_response_text(resp)
#             return txt
#         except Exception as e:
#             last_exc = e
#             attempt += 1
#             if attempt <= retries:
#                 time.sleep(backoff_s * attempt)
#             else:
#                 raise RuntimeError(f"Vision model call failed after {retries+1} attempts: {e}") from e
#     raise RuntimeError(f"Vision model call failed: {last_exc}")


# def merge_extras_into_plain_text_json(
#     plain_text_json_path: Path,
#     extras_text_path: Path,
#     out_json_path: Path,
#     title: Optional[str] = None,
# ):
#     """
#     Simple merger: read existing Plain_Text.json (list of sections) if exists,
#     append a new section with title 'Multimodal Extras' (or provided title) whose text
#     is contents of extras_text_path.
#     """
#     extras = extras_text_path.read_text(encoding="utf-8")
#     new_section = {
#         "title": title or "Multimodal Extras",
#         "text": extras,
#         "start_page": None,
#         "end_page": None,
#     }

#     if plain_text_json_path and plain_text_json_path.exists():
#         try:
#             sections = json.loads(plain_text_json_path.read_text(encoding="utf-8"))
#             if not isinstance(sections, list):
#                 sections = []
#         except Exception:
#             sections = []
#     else:
#         sections = []

#     sections.append(new_section)
#     out_json_path.parent.mkdir(parents=True, exist_ok=True)
#     out_json_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
#     return out_json_path


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pdf", required=True, help="Input PDF path")
#     ap.add_argument("--out_prefix", required=True, help="Output prefix (files will be <prefix>_extras.*)")
#     ap.add_argument("--model", default="gpt-4.1", help="Vision model (gpt-4.1 or gpt-5.1)")
#     ap.add_argument("--dpi", type=int, default=200, help="Render DPI for page images")
#     ap.add_argument("--max_pages", type=int, default=0, help="0=all pages, else limit")
#     ap.add_argument("--start_page", type=int, default=1, help="1-based start page")
#     ap.add_argument("--end_page", type=int, default=0, help="1-based end page (0 = last page)")
#     ap.add_argument("--max_output_tokens", type=int, default=1500, help="Max tokens per page output")
#     ap.add_argument("--no_machine_text", action="store_true", help="Do not include machine-extracted text in prompt")
#     ap.add_argument("--merge_plain_text_json", type=str, default="", help="Path to existing Plain_Text.json to append extras into")
#     ap.add_argument("--merged_out_json", type=str, default="Plain_Text_with_extras.json", help="Path to write merged Plain_Text.json if --merge_plain_text_json provided")
#     args = ap.parse_args()

#     pdf_path = Path(args.pdf)
#     if not pdf_path.exists():
#         print("PDF not found:", pdf_path, file=sys.stderr)
#         sys.exit(1)

#     if not os.getenv("OPENAI_API_KEY"):
#         print("Missing OPENAI_API_KEY environment variable.", file=sys.stderr)
#         sys.exit(1)

#     out_prefix = Path(args.out_prefix)
#     out_txt = out_prefix.with_suffix("") .with_name(out_prefix.name + "_extras.txt")
#     out_jsonl = out_prefix.with_suffix("") .with_name(out_prefix.name + "_extras.jsonl")

#     client = OpenAI()  # uses OPENAI_API_KEY from env

#     doc = fitz.open(str(pdf_path))
#     n_pages = doc.page_count
#     start = max(1, args.start_page)
#     end = args.end_page if args.end_page and args.end_page > 0 else n_pages
#     end = min(end, n_pages)
#     pages = list(range(start, end + 1))
#     if args.max_pages and args.max_pages > 0:
#         pages = pages[: args.max_pages]

#     print(f"Processing PDF: {pdf_path.name}  pages: {n_pages}  -> processing: {pages[0]}..{pages[-1]}")

#     out_txt.parent.mkdir(parents=True, exist_ok=True)
#     with out_txt.open("w", encoding="utf-8") as ftxt, out_jsonl.open("w", encoding="utf-8") as fjsonl:
#         for p in pages:
#             page = doc.load_page(p - 1)
#             machine_text = "" if args.no_machine_text else extract_machine_text(page)
#             png_bytes = render_page_to_png_bytes(page, dpi=args.dpi)
#             data_url = png_bytes_to_data_url(png_bytes)

#             try:
#                 extras = call_vision_model(
#                     client=client,
#                     model=args.model,
#                     machine_text=machine_text,
#                     image_data_url=data_url,
#                     max_output_tokens=args.max_output_tokens,
#                 )
#             except Exception as e:
#                 extras = f"[ERROR] page {p}: {e}"
#                 print("ERROR calling vision model on page", p, e)

#             header = f"\n\n==================== PAGE {p} ====================\n"
#             ftxt.write(header)
#             ftxt.write(extras + "\n")
#             ftxt.flush()

#             rec = {
#                 "pdf": str(pdf_path),
#                 "page": p,
#                 "model": args.model,
#                 "extras_text": extras,
#                 "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#             }
#             fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
#             fjsonl.flush()

#             print(f"Page {p}: {'OK' if not extras.startswith('[ERROR]') else 'ERROR'}")

#     print(f"\nWrote:\n - {out_txt}\n - {out_jsonl}")

#     # Optional: merge into Plain_Text.json
#     if args.merge_plain_text_json:
#         in_json = Path(args.merge_plain_text_json)
#         out_json = Path(args.merged_out_json)
#         merged = merge_extras_into_plain_text_json(
#             plain_text_json_path=in_json,
#             extras_text_path=out_txt,
#             out_json_path=out_json,
#             title="Multimodal Extras (vision-derived)"
#         )
#         print("Merged extras into:", merged)





# from pathlib import Path
# import subprocess

# # --- paths ---
# PDF_PATH = "Shortest_test.pdf"          # path to your PDF
# OUT_PREFIX = "MultiModaloutputs/6578"         # output prefix (no extension)
# MODEL = "gpt-4.1"                   # or "gpt-5.1"

# # make sure output directory exists
# Path("MultiModaloutputs").mkdir(exist_ok=True)
# # --- run the script ---
# cmd = [
#     "python", "pdf_multimodal_extras.py",
#     "--pdf", PDF_PATH,
#     "--out_prefix", OUT_PREFIX,
#     "--model", MODEL,
#     "--max_pages", "5"   # set 0 to process all pages
# ]

# print("Running:\n", " ".join(cmd))
# subprocess.run(cmd, check=True)



# !pip install pymupdf

# Single-cell runner for PDF -> multimodal extras (no subprocess)
# Requires: pip install pymupdf openai
import os, time, base64, json
from pathlib import Path
import fitz  # pymupdf
from openai import OpenAI

# ======== USER CONFIG ========
PDF_PATH = "Shortest_test2.pdf"                   # path to your PDF file
OUT_PREFIX = "MultiModaloutputs/output/6578"            # prefix for outputs (will create files like ..._extras.txt)
MODEL = "gpt-5.1"  #"gpt-4.1"                                # or "gpt-5.1"
DPI = 200
MAX_PAGES = 5          # 0 = all pages
START_PAGE = 1
END_PAGE = 0           # 0 means last page
MAX_OUTPUT_TOKENS = 1500
INCLUDE_MACHINE_TEXT = True   # set False to not include page-extracted text in prompt
# ===========================

# sanity checks
if not Path(PDF_PATH).exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in environment. Use dotenv or set env var before running.")

# utilities (same logic as the script)
def render_page_to_png_bytes(page: fitz.Page, dpi: int = 200) -> bytes:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

def extract_machine_text(page: fitz.Page, max_chars: int = 12000) -> str:
    txt = page.get_text("text") or ""
    txt = txt.strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n[TRUNCATED]"
    return txt

# SYSTEM_INSTRUCTIONS = (
#     "You are a precise document-vision assistant. You will receive MACHINE_TEXT (plain text extracted programmatically) "
#     "and PAGE_IMAGE (the page rendered as an image). Output ONLY the visual content missing from MACHINE_TEXT: "
#     "TABLES, FIGURES/DIAGRAMS, EQUATIONS/FORMULAS, CHARTS/PLOTS, TRUTH_TABLES. "
#     "If nothing missing output exactly: NO_NON_TEXT_ELEMENTS. Use tags: [TABLE],[FIGURE],[EQUATION],[CHART],[DIAGRAM],[TRUTH_TABLE]. "
#     "Keep each block short and sentence-friendly."
# )

# SYSTEM_INSTRUCTIONS ="""
# You are a semantic extractor for a KNOWLEDGE GRAPH pipeline.

# You will be given:
# 1) MACHINE_TEXT: plain text already extracted from the page
# 2) PAGE_IMAGE: a rendered image of the same page

# CRITICAL CONTEXT
# The output of this step will be fed DIRECTLY into an automated Knowledge Graph (KG)
# generation pipeline. Therefore:
# - Do NOT produce narrative or descriptive text.
# - Do NOT describe visual layout, arrows, boxes, or diagram aesthetics.
# - Do NOT write for humans.
# - Write ONLY semantic, normalized, KG-ready statements.

# YOUR TASK
# Extract ONLY information that is present in PAGE_IMAGE but is NOT already fully captured
# in MACHINE_TEXT, and express it as SEMANTIC FACTS suitable for KG triplet extraction.

# TARGET OUTPUT STYLE (MANDATORY)
# Produce short, atomic, declarative statements that:
# - Can be directly converted into (subject, relation, object) triplets
# - Use canonical concept names (no visual language like “top box”, “arrow”, “diagram shows”)
# - Express hierarchy, containment, equivalence, temporal scope, or functional relations explicitly

# AVOID COMPLETELY
# - “This figure shows…”
# - “The diagram illustrates…”
# - “Arrows indicate…”
# - Any reference to visual position, layout, or formatting

# ALLOWED RELATION TYPES (USE THESE IDEAS)
# You are NOT limited to these labels, but your statements should eventually become triplets with relations such as(or any reacher relation):
# - is_a
# - part_of
# - consists_of
# - subset_of
# - includes
# - excludes
# - equals
# - measured_over
# - derived_from
# - defined_as
# - requires
# - depends_on
# - spans_time
# - contained_in
# - aggregated_into

# OUTPUT FORMAT
# If no missing non-text information exists, output EXACTLY:
# NO_NON_TEXT_ELEMENTS

# Otherwise, output blocks using ONE of the following headers:
# [FIGURE_FACTS]
# [TABLE_FACTS]
# [EQUATION_FACTS]
# [CHART_FACTS]
# [DIAGRAM_FACTS]
# [TRUTH_TABLE_FACTS]

# Inside each block:
# - Use BULLET POINTS only
# - Each bullet must contain ONE atomic semantic fact
# - Prefer one relation per bullet
# - Do NOT combine multiple relations in a single sentence

# NORMALIZATION RULES
# - Use singular noun phrases for concepts (e.g., “Productive Time”, not “productive times”)
# - Use consistent naming across bullets
# - If two concepts are equivalent, state it explicitly
# - If a concept aggregates others, list them via multiple bullets

# EXAMPLES (ILLUSTRATIVE ONLY)

# BAD:
# “The diagram shows a hierarchy of equipment time states.”

# GOOD:
# [FIGURE_FACTS]
# - Total Time includes Nonscheduled Time
# - Total Time includes Operations Time
# - Operations Time includes Engineering Time
# - Operations Time includes Manufacturing Time
# - Manufacturing Time includes Productive Time
# - Manufacturing Time includes Standby Time

# EQUATIONS
# If equations appear visually:
# - Provide symbolic form (LaTeX if possible)
# - Then provide semantic definitions as facts (not prose)

# GOOD:
# [EQUATION_FACTS]
# - MTBF_u equals uptime divided by number_of_failures_during_uptime
# - MTBF_u measures mean_time_between_failures_during_uptime
# - uptime is a duration_of Equipment Uptime

# TABLES
# For tables:
# - Convert rows into factual statements
# - Do NOT restate column headers narratively

# GOOD:
# [TABLE_FACTS]
# - MTBF_u has_formula uptime / failures_during_uptime
# - MTBF_u has_reference_section 8.2.1.1

# FINAL CHECK (IMPORTANT)
# Before responding, verify:
# - Every bullet could reasonably become a KG edge
# - No bullet depends on visual wording
# - No bullet is purely descriptive without semantic value

# """

SYSTEM_INSTRUCTIONS ="""
You convert non-text elements in technical PDF pages (figures, tables, equations, charts, diagrams)
into a compact, standalone TEXT SURROGATE that preserves the full meaning of the element.

CRITICAL:
This text will be fed into an automated Knowledge Graph pipeline later.
Therefore, DO NOT output KG-like triples and DO NOT use predicate tokens such as:
"is_a", "includes", "part_of", "subset_of", "equals" as standalone relation statements.

Write like a standards/specification document: concise, explicit, information-complete.

INPUTS:
1) MACHINE_TEXT: plain text already extracted from the page.
2) PAGE_IMAGE: a rendered image of the same page.

TASK:
Extract ONLY the information conveyed by non-text elements in PAGE_IMAGE that is not already
fully present in MACHINE_TEXT (tables, figures, diagrams, charts, equations, truth tables).
Convert each element into a self-contained textual description that would still be useful
if the image were removed.

DO NOT:
- Do NOT describe layout or visual features (no "box", "arrow", "top", "bottom", "left", "right", "stack", "diagram shows").
- Do NOT narrate ("this figure shows", "illustrates", "depicts").
- Do NOT invent new named concepts that do not appear in the element text/caption.
- Do NOT create taxonomy statements like "X is a Y" unless explicitly written in the element itself.

OUTPUT FORMAT (MANDATORY):
If there is no non-text information missing, output exactly:
NO_NON_TEXT_ELEMENTS

Otherwise output one or more blocks. Each block MUST start with exactly one of:
[FIGURE]
[TABLE]
[EQUATION]
[CHART]
[DIAGRAM]
[TRUTH_TABLE]

Inside each block use the following template:

Identifier: (e.g., Figure 1 / Table 3 / Equation (4) if visible; otherwise "Unknown")
Caption/Title: (verbatim if visible; otherwise omit)

Text surrogate:
- Write 4–12 short sentences (not bullets) that fully capture the element’s meaning.
- Prefer explicit decomposition statements and compact math-style definitions.
- Use consistent term names exactly as written in the element (case-insensitive normalization is OK).
- When the element defines a hierarchy or breakdown, express it using short “definition equations”
  like: "Total Time = Nonscheduled Time + Equipment Downtime + Equipment Uptime."
- When the element lists categories, list them explicitly.
- If a table appears: reproduce it as Markdown table, then add 2–6 sentences summarizing what each row means.
- If an equation appears: output LaTeX (or plain-text math), then 2–6 sentences defining variables as stated.

QUALITY BAR:
The text surrogate should be sufficient for a reader to reconstruct the meaning of the element
without seeing the image, with no reliance on layout language.

Keep it compact and factual.

"""


def parse_response_text(resp) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    out = ""
    if hasattr(resp, "output") and resp.output:
        for item in resp.output:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            out += c.get("text","")
                        elif isinstance(c, str):
                            out += c
                elif isinstance(content, str):
                    out += content
            elif isinstance(item, str):
                out += item
    if out:
        return out.strip()
    return str(resp)

def call_vision_model(client: OpenAI, model: str, machine_text: str, image_data_url: str, max_output_tokens: int = 1500):
    # user_prompt = (
    #     "MACHINE_TEXT:\n```\n" + (machine_text if machine_text else "[EMPTY]") + "\n```\n\n"
    #     "Now inspect the PAGE_IMAGE and output ONLY what is missing from MACHINE_TEXT. "
    #     "If nothing missing write exactly: NO_NON_TEXT_ELEMENTS. Use block tags as requested."
    # )
    user_prompt = (
    "MACHINE_TEXT (plain text already extracted from this page):\n"
    "```\n"
    + (machine_text if machine_text else "[EMPTY]")
    + "\n```\n\n"
    "Now inspect the PAGE_IMAGE and extract ONLY the information conveyed by non-text elements "
    "(figures, tables, diagrams, charts, equations) that is NOT already fully present in MACHINE_TEXT.\n\n"
    "IMPORTANT OUTPUT REQUIREMENTS:\n"
    "- Produce a standalone TEXT SURROGATE written in technical/specification style.\n"
    "- The text must fully explain the meaning of the visual element as if the image were removed.\n"
    "- Do NOT output KG triples, pseudo-relations, or predicate-style statements.\n"
    "- Do NOT use words like: is_a, includes, part_of, subset_of, equals.\n"
    "- Do NOT mention visual layout or structure (no boxes, arrows, top/bottom, left/right, stack, diagram shows).\n"
    "- Do NOT narrate (avoid phrases like 'this figure shows' or 'the diagram illustrates').\n\n"
    "FORMAT:\n"
    "- If no missing non-text information exists, output exactly: NO_NON_TEXT_ELEMENTS\n"
    "- Otherwise, output one or more blocks starting with exactly one of:\n"
    "  [FIGURE], [TABLE], [EQUATION], [CHART], [DIAGRAM], [TRUTH_TABLE]\n"
    "- Inside each block, write 4–12 short declarative sentences that fully capture the semantics.\n"
    "- Prefer compact definition-style sentences and decomposition equations where appropriate.\n"
)

    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[{
            "role":"user",
            "content": [
                {"type":"input_text","text": user_prompt},
                {"type":"input_image","image_url": image_data_url}
            ]
        }],
        max_output_tokens=max_output_tokens,
    )
    return parse_response_text(resp)

# prepare outputs
out_prefix = Path(OUT_PREFIX)
out_prefix.parent.mkdir(parents=True, exist_ok=True)
out_txt = out_prefix.with_name(out_prefix.name + "_extras.txt")
out_jsonl = out_prefix.with_name(out_prefix.name + "_extras.jsonl")

client = OpenAI()  # uses OPENAI_API_KEY from env
doc = fitz.open(PDF_PATH)
n_pages = doc.page_count
start = max(1, START_PAGE)
end = END_PAGE if (END_PAGE and END_PAGE>0) else n_pages
end = min(end, n_pages)
pages = list(range(start, end+1))
if MAX_PAGES and MAX_PAGES>0:
    pages = pages[:MAX_PAGES]

print(f"PDF: {PDF_PATH} | pages in file: {n_pages} | processing pages: {pages[0]}..{pages[-1]}")

with out_txt.open("w", encoding="utf-8") as ftxt, out_jsonl.open("w", encoding="utf-8") as fjsonl:
    for p in pages:
        page = doc.load_page(p-1)
        machine_text = "" if not INCLUDE_MACHINE_TEXT else extract_machine_text(page)
        png_bytes = render_page_to_png_bytes(page, dpi=DPI)
        data_url = png_bytes_to_data_url(png_bytes)

        try:
            extras = call_vision_model(
                client=client,
                model=MODEL,
                machine_text=machine_text,
                image_data_url=data_url,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
        except Exception as e:
            extras = f"[ERROR] page {p}: {e}"
            print("Error on page", p, e)

        header = f"\n\n==================== PAGE {p} ====================\n"
        ftxt.write(header)
        ftxt.write(extras + "\n")
        ftxt.flush()

        rec = {"pdf": str(PDF_PATH), "page": p, "model": MODEL, "extras_text": extras, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fjsonl.flush()

        print(f"Page {p}: {'OK' if not extras.startswith('[ERROR]') else 'ERROR'}")

print("Done. Wrote:", out_txt, out_jsonl)
print("--- Preview (first 1000 chars) ---")
print(out_txt.read_text(encoding="utf-8")[:1000])
