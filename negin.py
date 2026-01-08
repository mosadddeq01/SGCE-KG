# %%
# pip install kg-gen

!pip install tf-keras

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

# %%
import os, nltk

# find an nltk data base path that exists and create the expected file
created = False
for base in nltk.data.path:
    target = os.path.join(base, "tokenizers", "punkt", "PY3_tab")
    if os.path.exists(target):
        print("Already present:", target)
        created = True
        break

if not created:
    # choose the first nltk data path (nltk will search these)
    base = nltk.data.path[0]
    folder = os.path.join(base, "tokenizers", "punkt")
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "PY3_tab"), "a").close()
    print("Created:", os.path.join(folder, "PY3_tab"))

#import this file: Experiments/KG-Gen/kg_gen_cloned/src/kg_gen/kg_gen.py
import sys
sys.path.append(
   "Experiments/KG-Gen/kg_gen_cloned/src/kg_gen"
)
from kg_gen import KGGen


# %%




#########################
import sys
sys.path.append(
   "Experiments/KG-Gen/kg_gen_cloned/src"
)
from kg_gen import KGGen
#########################


# import dotenv and load api from .env 
import dotenv
import os
dotenv.load_dotenv(".env")




# # Initialize KGGen with optional configuration
# kg = KGGen(
# #   model="openai/gpt-4",  # Default model
# #   temperature=0.0,        # Default temperature
#   reasoning_effort="high",
#   api_key=os.getenv("OPENAI_API_KEY"),
# )


kg = KGGen(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# EXAMPLE 1: Single string with context
# text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father.Alis is John's mom.Josh's mother is a friend of Alis."
text_input = "```4.2.3.6 Prevention / Mitigation\na) Existing Materials\ni) Temper embrittlement cannot be prevented if the material contains critical levels of the embrittling\nimpurity elements and is exposed in the embrittling temperature range.\nii) To minimize the possibility of brittle fracture during startup and shutdown, many refiners use a\npressurization sequence to limit system pressure to about 25 percent of the maximum design\npressure for temperatures below a Minimum Pressurization Temperature (MPT).\niii) MPT’s generally range from 350oF (171oC) for the earliest, most highly temper embrittled steels,\ndown to 150oF (38oC) or lower for newer, temper embrittlement resistant steels (as required to also\nminimize effects of hydrogen embrittlement).\niv) If weld repairs are required, the effects of temper embrittlement can be temporarily reversed (de-\nembrittled) by heating at 1150°F (620°C) for 2 hours per inch of thickness, and rapidly cooling to\nroom temperature. It is important to note that re-embrittlement will occur over time if the material is\nre-exposed to the embrittling temperature range.\nb) New Materials\ni) The best way to minimize the likelihood and extent of temper embrittlement is to limit the\nacceptance levels of manganese, silicon, phosphorus, tin, antimony, and arsenic in the base metal\nand welding consumables. In addition, strength levels and PWHT procedures should be specified\nand carefully controlled.\nii) A common way to minimize temper embrittlement is to limit the \"J*\" Factor for base metal and the\n\"X\" Factor for weld metal, based on material composition as follows:\nJ* = (Si + Mn) x (P + Sn) x 104 {elements in wt%}\nX =(10P + 5Sb + 4Sn + As)/100 {elements in ppm}\niii) Typical J* and X factors used for 2.25 Cr steel are 100 and 15, respectively. Studies have also\nshown that limiting the (P + Sn) to less than 0.01% is sufficient to minimize temper embrittlement\nbecause (Si + Mn) control the rate of embrittlement.\niv) Expert metallurgical advice should be solicited to determine acceptable composition, toughness\nand strength levels, as well as appropriate welding, fabricating and heat treating procedures for\nnew low alloy steel heavy wall equipment and low alloy equipment operating in the creep range.\n4.2.3.7 Inspection and Monitoring\na) A common method of monitoring is to install blocks of original heats of the alloy steel material inside the\nreactor. Samples are periodically removed from these blocks for impact testing to monitor progress of\ntemper embrittlement or until a major repair issue arises.\nb) Process conditions should be monitored to ensure that a proper pressurization sequence is followed to\nhelp prevent brittle fracture due to temper embrittlement.\n4.2.3.8 Related Mechanisms\nNot applicable.\n4.2.3.9 References\n1. R.A. Swift , “Temper Embrittlement in Low Alloy Ferritic Steels,” CORROSION/76, Paper #125, NACE,\n1976.\n2. R.A. White and E.F. Ehmke, “Materials Selection for Refineries and Associated Facilities,” National\nAssociation of Corrosion Engineers, NACE, 1991, pp. 53-54.\n3. R. Viswanathan, “Damage Mechanisms and Life Assessment of High Temperature Components,” ASM\nInternational, 1989.\n4-10 API Recommended Practice 571 December 2003\n4. API Recommended Practice 934, Materials and Fabrication Requirements for 2-1/4 Cr-1Mo and 3Cr-\n1Mo Steel Heavy Wall Pressure Vessels for High Temperature, High Pressure Service, American\nPetroleum Institute, Washington, D.C.\nDecember 2003 API Recommended Practice 571 4-11\nFigure 4-5 – Plot of CVN toughness as a function of temperature showing a shift in the 40-ft-lb\ntransition temperature.```",

graph_1 = kg.generate(
  input_data=text_input,
#   context="Family relationships",
)


# Output: 
# entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# edges={'is brother of', 'is father of', 'is mother of'} 
# relations={('Ben', 'is brother of', 'Josh'), 
#           ('Andrew', 'is father of', 'Josh'), 
#           ('Linda', 'is mother of', 'Josh')}

print("Entities:", graph_1.entities)
print("Relations:", graph_1.relations)




KGGen.visualize(
    graph_1,
    output_path="kg_example_4.2.3.6.html",
    open_in_browser=True
)



























# %%
# Example-ready script for KGGen: chunking + clustering + other examples
# Save this as run_kggen_examples.py or run in a notebook cell.
from kg_gen import KGGen
import os
from pathlib import Path
import dotenv 

# --- CONFIG: set your key safely (preferred) ---
# Option A (recommended): set in your shell before running:
# export OPENAI_API_KEY="sk-..."
# Option B (quick): set here (but DON'T paste key into shared chats)
os.environ["OPENAI_API_KEY"] = dotenv.get_key(".env", "OPENAI_API_KEY")

# Initialize KGGen
kg = KGGen(
    model="openai/gpt-4o",
    temperature=0.0
    # api_key argument optional if OPENAI_API_KEY is set in env
)

# Ensure output directory is writable in this environment
from pathlib import Path
outdir = Path.home() / "/Users/nesmaei2/ASU Dropbox/Negin Esmaeili/NeginDB/NeMoBS/NE MO/MO Paper"
outdir.mkdir(parents=True, exist_ok=True)

# ---------------------
# Example 2: chunking + clustering on large text
# ---------------------
large_text_path = outdir / "large_text.txt"

# If you don't already have a large text file, write a small sample so the example runs.
if not large_text_path.exists():
    sample = """Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
that uses multiple layers of neural networks. Supervised learning requires training data to learn
patterns. Machine learning is a type of AI technology that enables computers to learn from data.
AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
many fields of study. Unsupervised learning finds structure without labeled data. Reinforcement learning
optimizes policies by reward signals. Deep neural networks often require large datasets and GPU compute.
"""
    large_text_path.write_text(sample, encoding="utf-8")

large_text = large_text_path.read_text(encoding="utf-8")

# Run KGGen with chunking and clustering
graph_2 = kg.generate(
    input_data=large_text,
    chunk_size=5000,   # chunk-size in chars
    cluster=True       # ask the library to cluster synonymous entities/edges
)

# Inspect results in Python
print("Entities:", getattr(graph_2, "entities", None))
print("Relations:", getattr(graph_2, "relations", None))
print("Entity clusters (if present):", getattr(graph_2, "entity_clusters", None))
print("Edge clusters (if present):", getattr(graph_2, "edge_clusters", None))

# Visualize and save HTML into /mnt/data
html_out = outdir / "kg_example2.html"
KGGen.visualize(graph_2, output_path=str(html_out), open_in_browser=False)
print("Saved visualization to:", html_out)

# ---------------------
# Example 3: messages array
# ---------------------
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages)
print("Example 3 entities:", getattr(graph_3, "entities", None))
KGGen.visualize(graph_3, output_path=str(outdir / "kg_example3.html"), open_in_browser=False)

# ---------------------
# Example 4: combine graphs + clustering
# ---------------------
text1 = "Linda is Joe's mother. Ben is Joe's brother."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

g_a = kg.generate(input_data=text1)
g_b = kg.generate(input_data=text2)

combined = kg.aggregate([g_a, g_b])
# Optionally cluster the combined graph (resolve Joe vs Joseph)
clustered_combined = kg.cluster(combined, context="Family relationships")

print("Combined entities:", getattr(clustered_combined, "entities", None))
KGGen.visualize(clustered_combined, output_path=str(outdir / "kg_combined_clustered.html"), open_in_browser=False)
print("Saved combined visualization to:", outdir / "kg_combined_clustered.html")

# Done
print("All done. Check /mnt/data for HTML visualizations.")


# %%


# %%
# Save sample & viz next to the original JSONL (copy/paste into a Jupyter cell)
import os, json, traceback
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML

os.environ["OPENAI_API_KEY"] = "sk-svcacct-nPW5ufSOF8XAU00GTHlafXNnn6TQOOI0DMbTFnl94sIFILngTS2d0b8mEwz-p8r1xT3BlbkFJ8I7u_wrq2TyB6GPDmTGCaSLgXRTTv1avc7TY_gX7kMC3z8R6QkUEuEX9XIv5mw2AA"


# ---------- CONFIG ----------
orig_path = Path("/Users/nesmaei2/ASU Dropbox/Negin Esmaeili/NeginDB/NeMoBS/NE MO/MO Paper/chunks_sentence.jsonl")
N = 20  # number of records to sample (change to 20 if you want)
# filenames to create next to orig file:
sample_fname = "sample_chunks.jsonl"
html_fname = "kg_viz_testmo.html"
# ----------------------------

# Check API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in your shell or notebook env before running.")

if not orig_path.exists():
    raise FileNotFoundError(f"Original file not found at {orig_path}. Check the path.")

out_dir = orig_path.parent
sample_path = out_dir / sample_fname
html_out = out_dir / html_fname

# Read first N JSONL records
sample_items = []
with orig_path.open("r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= N:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            obj = {"text": line}
        sample_items.append(obj)

# Try to write sample JSONL next to original
try:
    with sample_path.open("w", encoding="utf-8") as fout:
        for item in sample_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample_items)} sample records to: {sample_path}")
except Exception as e:
    print("Failed to write sample file to same folder. Error:")
    traceback.print_exc()
    # fallback to /tmp if cannot write in original folder
    sample_path = Path("/tmp") / sample_fname
    with sample_path.open("w", encoding="utf-8") as fout:
        for item in sample_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved sample to fallback path: {sample_path}")

# Show a neat table preview
def make_row(obj):
    return {
        "id": obj.get("id"),
        "ref_title": obj.get("ref_title"),
        "start_page": obj.get("start_page"),
        "end_page": obj.get("end_page"),
        "chunk_index_in_section": obj.get("chunk_index_in_section"),
        "n_words": obj.get("n_words"),
        "n_tokens_est": obj.get("n_tokens_est"),
        "text_snippet": (obj.get("text") or obj.get("content") or str(obj))[:300].replace("\n"," "),
    }

df = pd.DataFrame([make_row(o) for o in sample_items])
display(df)

# ---------- Run KGGen on the small sample ----------
try:
    from kg_gen import KGGen
except Exception as e:
    raise RuntimeError("kg_gen not installed or import failed. Run `pip install kg-gen` in your environment.") from e

kg = KGGen(model="openai/gpt-4o", temperature=0.0, api_key=api_key)

graphs = []
for rec in sample_items:
    try:
        if isinstance(rec, dict) and "messages" in rec and isinstance(rec["messages"], list):
            g = kg.generate(input_data=rec["messages"], cluster=False)
        elif isinstance(rec, dict) and ("text" in rec or "content" in rec):
            txt = rec.get("text") or rec.get("content")
            g = kg.generate(input_data=txt, chunk_size=2000, cluster=False)
        elif isinstance(rec, str):
            g = kg.generate(input_data=rec, chunk_size=2000, cluster=False)
        else:
            g = kg.generate(input_data=json.dumps(rec), chunk_size=2000, cluster=False)
        graphs.append(g)
        print("Processed sample record; entities:", getattr(g, "entities", None))
    except Exception as e:
        print("kg.generate failed for one record:", e)
        traceback.print_exc()

# Aggregate + cluster
combined = kg.aggregate(graphs)
clustered = kg.cluster(combined, context="Mohammad research small test")

print("Final entities count:", len(getattr(clustered, "entities", [])))
print("Some relations (sample):", list(getattr(clustered, "relations", []))[:20])

# Visualization: attempt to save into same folder as original
try:
    KGGen.visualize(clustered, output_path=str(html_out), open_in_browser=False)
    print("Saved visualization HTML to:", html_out)
except Exception as e:
    print("Failed to write visualization into same folder. Error:")
    traceback.print_exc()
    # fallback to /tmp
    fallback_html = Path("/tmp") / html_fname
    try:
        KGGen.visualize(clustered, output_path=str(fallback_html), open_in_browser=False)
        html_out = fallback_html
        print("Saved visualization to fallback path:", html_out)
    except Exception as e2:
        print("Also failed to write fallback visualization file:", e2)
        raise

# Display visualization inside notebook if possible
try:
    display(HTML(f'<iframe src="file://{html_out}" width="100%" height="600"></iframe>'))
except Exception as e:
    print("Could not embed the HTML preview. You can open this file in your file browser or a notebook cell with HTML.")
    print("Visualization location:", html_out)


# %%



