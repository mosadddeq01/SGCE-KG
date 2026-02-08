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
