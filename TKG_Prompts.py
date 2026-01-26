import json

#!############################################# Start Chapter ##################################################
#region:#!   Second Version



#?######################### Start ##########################
#region:#?   Entity Recognition Prompt


# -------------------------
# Entity Recognition Prompt
# -------------------------

ENT_REC_PROMPT_TEMPLATE = """GOAL: We are building a context-enriched knowledge graph (KG) from text.

YOUR TASK (THIS STEP ONLY): Extract candidate ENTITY mentions from the FOCUS chunk ONLY.
- You may consult CONTEXT (previous chunks) ONLY for disambiguation (e.g., pronouns, acronyms).
- Do NOT extract relations here. Relations + qualifiers will be handled later in Relation Recognition (Rel Rec).

WHY THIS MATTERS (PIPELINE AWARENESS):
- Your outputs are later embedded, clustered, and resolved. The quality of:
  (1) entity_name, (2) entity_description, and (3) resolution_context
  strongly determines downstream canonicalization and schema quality.
- entity_type_hint is a lightweight signal only; keep it broad and reusable.

========================
HIGH-RECALL ENTITY POLICY
========================
- Prefer recall over precision. If a candidate mention seems potentially useful, include it.
- Avoid obvious noise tokens (e.g., “this”, “it”, “the system”) unless they clearly refer to a real entity
  and you can disambiguate it.

========================
ENTITY NAMING RULES (CRITICAL)
========================
1) entity_name is an ENTITY-LEVEL label (mention-normalized), NOT a schema class.
   - GOOD entity_name examples (instance or concept): “centrifugal pump P-101”, “inflation”, “Pride and Prejudice”.
   - BAD entity_name examples (schema-ish): “Pump”, “Failure Mechanism”, “Organization”.

2) Prefer SHORT, STABLE, REUSABLE labels:
   - For recurring concepts (phenomena, processes, behaviors, conditions, methods), use a reusable name.
   - If the text is indirect (“this type of…”, “loss of…”, “a pattern of…”), infer the best short label and justify it via resolution_context.

3) Deduplicate within the FOCUS chunk:
   - If the same entity is mentioned multiple times in the focus chunk, output ONE object for it
     (choose the best proving context_phrase).
   - If the same surface string refers to clearly different senses, output separate objects with disambiguated names.

========================
INTRINSIC NODE PROPERTIES (MUST DO)
========================
Use node_properties ONLY for intrinsic attributes of the entity itself (NOT relations).

Intrinsic properties include (examples are illustrative):
- identifiers/codes: serial_number, asset_id, standard_id
- dates/years: publication_year, installation_date, manufacture_year
- numeric capacities/ratings: max_temperature, rated_pressure, power_rating
- characteristics/features: phase, shape, structure, color
- specification details: material_grade, alloy_type, design_class
- performance metrics: efficiency, durability, strength

Rules:
- If a fact does NOT naturally involve TWO DISTINCT entities, it belongs in node_properties.
- Do NOT encode relation qualifiers here (e.g., “at high temperature”, “during election season”, “under policy X”).
  Those belong to Relation Recognition.

node_properties format:
- Only include node_properties if at least one intrinsic property is explicitly present in the FOCUS chunk.
- Each property object must be: {{ "prop_name": "...", "prop_value": "...", "justification": "..." }}
- prop_name should be short and stable (prefer snake_case).
- prop_value may be a string or number (include units inside the string when relevant).

========================
CONFIDENCE GUIDELINES
========================
- 0.90–1.00 : Explicit and unambiguous in FOCUS.
- 0.70–0.89 : Strongly supported; minor disambiguation via CONTEXT.
- 0.40–0.69 : Plausible inference; partial support (still include).
- 0.00–0.39 : Weakly supported; include only if likely useful.

========================
TYPE-HINT GUIDANCE (OPEN-WORLD, BROAD ONLY)
========================
entity_type_hint is a very broad “family / category” hint for the entity.
It is NOT a strict ontology label and is NOT required to come from any fixed list.

Rules:
- Provide ONE concise token (ideally one word / no spaces). Examples: Person, Organization, Location, Document,
  Artifact, Component, Material, Event, Process, Condition, Method, Concept, Role, Property, Measurement, Policy.
  (These are examples only — not exhaustive.)
- You MAY invent a new token when needed, but keep it broad and reusable across many documents.
- Do NOT encode instance identifiers, dates, or ultra-specific subtypes into entity_type_hint.
  BAD: “Pump_P-101”, “JazzSubgenre_Bebop”, “ValveSeatWearMechanism_2020”
  GOOD: “Component", “Mechanism”, “Condition”
- If unsure, use a conservative broad token (e.g., “Concept” or “Other”) and lower confidence_score.

========================
OUTPUT FORMAT (STRICT — REQUIRED)
========================
Return ONLY a single JSON array (no commentary, no markdown fences).

Each array element MUST be an object with EXACTLY these keys:
- "entity_name" (string)
- "entity_description" (string) — 10–25 words, derived from FOCUS (and CONTEXT only if needed).
- "entity_type_hint" (string) — broad category token (open-world).
- "context_phrase" (string) — short 3–10 word excerpt from FOCUS proving the mention (empty string if impossible).
- "resolution_context" (string) — 20–120 words explaining why this mention maps to entity_name.
  Prefer the sentence containing the mention + at most one neighbor sentence; if CONTEXT was used, include at most one supporting sentence from CONTEXT.
- "confidence_score" (number) — 0.0–1.0
- "node_properties" (array, OPTIONAL) — include only when intrinsic properties are present in FOCUS.

IMPORTANT OUTPUT RULES:
- Extract entities that appear in the FOCUS chunk (not entities that appear only in CONTEXT).
- Do NOT output relations or relation qualifiers here.
- Use double quotes and valid JSON.
- If nothing is extractable, return [].

========================
CONTEXT (for disambiguation only)
========================
{context_block}

========================
FOCUS CHUNK (extract entities from here)
========================
FOCUS_CHUNK_ID: {focus_chunk_id}
{focus_text}

========================
EXAMPLE OUTPUT (3 diverse examples; JSON ARRAY ONLY)
========================
[
  {{
    "entity_name": "inflation",
    "entity_description": "Sustained rise in general price levels that reduces purchasing power over time.",
    "entity_type_hint": "Process",
    "context_phrase": "rising inflation",
    "resolution_context": "The focus text describes a continued increase in overall prices and its impact on purchasing power. This matches the standard economic process referred to as inflation.",
    "confidence_score": 0.86
  }},
  {{
    "entity_name": "centrifugal pump P-101",
    "entity_description": "A pump unit that moves fluid by converting rotational energy into pressure and flow.",
    "entity_type_hint": "Component",
    "context_phrase": "centrifugal pump P-101",
    "resolution_context": "The focus text names a specific pump with identifier P-101 and describes its fluid-moving function. This supports treating it as a concrete component entity.",
    "confidence_score": 0.93,
    "node_properties": [
      {{
        "prop_name": "rated_pressure",
        "prop_value": "10 bar",
        "justification": "The focus text states the pump’s rated pressure explicitly."
      }}
    ]
  }},
  {{
    "entity_name": "Pride and Prejudice",
    "entity_description": "A named written work referenced as a distinct novel within the discussed context.",
    "entity_type_hint": "Document",
    "context_phrase": "Pride and Prejudice",
    "resolution_context": "The focus text refers to 'Pride and Prejudice' as a specific titled work, indicating it is an identifiable document-level entity rather than a general concept.",
    "confidence_score": 0.88,
    "node_properties": [
      {{
        "prop_name": "publication_year",
        "prop_value": 1813,
        "justification": "The focus text explicitly provides the year associated with the work."
      }}
    ]
  }}
]
"""


#endregion#? Entity Recognition Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Entity Resolution Prompt

# -------------------------
# Entity Resolution Prompt
# -------------------------

ENT_RES_PROMPT_TEMPLATE = """ROLE: You are an entity-identity resolver for a knowledge graph.

SETTING (minimal context):
- A previous step extracted many candidate entity mentions from documents.
- This step resolves IDENTITY: it merges duplicates/aliases into canonical entities and cleans up noisy names/descriptions/type hints.
- Later steps (not shown here) will infer ontology-like classes and extract relations. Do NOT invent schema classes here.

IMPORTANT:
- The input group is created automatically (embedding-based clustering). It is only suggestive.
  It may contain:
  * true duplicates (same thing phrased differently),
  * near-duplicates (aliases/abbreviations),
  * or unrelated items that accidentally look similar.
  You must correct this by merging only when identity is truly the same.

=====================
INPUT YOU ARE GIVEN
=====================
A JSON array called "Group members".
Each member includes:
- id            : string (entity id)
- name          : string (current entity label; may be noisy)
- desc          : string (short description; may be noisy)
- type_hint     : string (broad role/family hint; may be inconsistent)
- confidence    : number or null
- text_span     : short excerpt that motivated this entity (may be an excerpt of context)
- chunk_text    : truncated chunk text (grounding evidence)

You MUST use ONLY this provided information.

=====================
YOUR TASK
=====================
Return an ORDERED JSON ARRAY of actions that improve entity identity quality:

1) MERGE duplicates:
   - Merge only when two or more members clearly refer to the SAME real-world entity
     OR the SAME conceptual entity (same concept, same meaning).
   - Common merge reasons:
     * abbreviations/acronyms (e.g., “mtDNA” vs “mitochondrial DNA”)
     * aliases/synonyms (e.g., “U.S.” vs “United States”)
     * spelling/format variants (e.g., “P-101” vs “P101”)
     * pronoun/placeholder entities that clearly refer to a specific member (e.g., “it”, “this unit”)

2) MODIFY noisy entries:
   - Improve an entity when the current name/desc/type_hint is misleading, overly-specific,
     too vague, or obviously a placeholder.
   - Examples of when to Modify:
     * name is a pronoun (“it”, “this”, “they”) but the referent is clear from chunk_text
     * name includes accidental context junk (e.g., leading “the”, trailing punctuation, etc.)
     * desc is not actually describing the entity (too relational, too contextual, or wrong)
     * type_hint is clearly wrong or inconsistent with others for the same entity

3) KEEP is implicit:
   - Any entity not mentioned in your actions will be kept unchanged.
   - You MAY include explicit KeepEntity actions, but it is OPTIONAL.

=====================
STRICT MERGE RULES
=====================
- Do NOT merge just because items co-occur in the same chunk.
- Do NOT merge just because type_hint is similar.
- Do NOT merge “related but different” entities:
  * instance vs class (unless the text clearly uses the class term to refer to that one instance)
  * parent vs subtype (e.g., “Jazz” vs “Bebop”)
  * concept vs measurement (e.g., “inflation” vs “3% inflation rate”)
- If the same surface string could refer to different things (“Apple” fruit vs company),
  do NOT merge; instead, MODIFY names to disambiguate if evidence supports it.

=====================
CANONICAL NAMING RULES
=====================
When you MERGE, choose:
- canonical_name: the BEST short, stable, reusable label for the unified entity.
  * Keep identity-critical identifiers when they are truly part of the entity identity
    (e.g., “P-101”, “ISO 9001”, “World War II”, “Album: Thriller”).
  * Remove purely local/accidental wording (e.g., “this”, “above”, “the following”).
- canonical_description: 1–2 sentences describing what the entity is, in a reusable way.
  Avoid encoding relations to other entities here.

=====================
TYPE HINT RULES (OPEN-WORLD)
=====================
type hints are broad role/family/category nudges used downstream for clustering and schema induction.
They must be:
- short (prefer 1 word; 2 words only if truly necessary)
- reusable (not tied only to this document)
- NOT an enumerated or instance-specific label

Examples (illustrative, NOT a fixed list):
Person, Organization, Location, Artifact, Work, Document, Event, Process, Method, Condition, Concept, Material, Component, Role, Measurement.

If unsure, keep the existing type_hint or choose a broader one, and explain briefly in rationale.

=====================
OUTPUT FORMAT (REQUIRED)
=====================
Return ONLY a JSON ARRAY (no markdown, no commentary).
Each element must be ONE of the following shapes:

A) MergeEntities
{{
  "action": "MergeEntities",
  "entity_ids": ["En_...", "En_..."],
  "canonical_name": "string",
  "canonical_description": "string",
  "canonical_type": "string",
  "rationale": "1–3 sentences grounded in provided text_span/chunk_text"
}}

B) ModifyEntity
{{
  "action": "ModifyEntity",
  "entity_id": "En_...",
  "new_name": "string or null",
  "new_description": "string or null",
  "new_type_hint": "string or null",
  "rationale": "1–3 sentences grounded in provided text_span/chunk_text"
}}

C) KeepEntity (OPTIONAL)
{{
  "action": "KeepEntity",
  "entity_id": "En_...",
  "rationale": "1 sentence"
}}

VALIDATION:
- Use ONLY ids that appear in the input group.
- Do NOT invent new entities.
- Do NOT output anything except the JSON array.

=====================
EXAMPLES (illustrative)
=====================

Example 1 (abbreviation merge):
Input members include:
- id=En_1 name="mtDNA"
- id=En_2 name="mitochondrial DNA"
Both chunk_text snippets clearly refer to the same molecule.

Output:

{{
  "action": "MergeEntities",
  "entity_ids": ["En_1","En_2"],
  "canonical_name": "mitochondrial DNA",
  "canonical_description": "DNA located in mitochondria, often discussed separately from nuclear DNA in genetics and biology.",
  "canonical_type": "Concept",
  "rationale": "Both members are the same concept: 'mtDNA' is a standard abbreviation for 'mitochondrial DNA' in the provided text."
}}


Example 2 (disambiguation via Modify, no merge):
Input members include:
- id=En_7 name="Apple"
- id=En_8 name="apple"
chunk_text shows En_7 is the company and En_8 is the fruit.

Output:

{{
  "action": "ModifyEntity",
  "entity_id": "En_7",
  "new_name": "Apple Inc.",
  "new_description": null,
  "new_type_hint": "Organization",
  "rationale": "Chunk text context indicates the company (products/business), not the fruit; renaming prevents accidental merges."
}}


Example 3 (engineering-style identifier merge, but domain-agnostic principle):
Input members include:
- id=En_10 name="P-101"
- id=En_11 name="pump P101"
chunk_text indicates both refer to the same tagged asset.

Output:

{{
  "action": "MergeEntities",
  "entity_ids": ["En_10","En_11"],
  "canonical_name": "pump P-101",
  "canonical_description": "A specific tagged pump unit referenced by identifier P-101 in the document.",
  "canonical_type": "Artifact",
  "rationale": "Both names refer to the same tagged unit; one is the bare tag and the other includes the role ('pump')."
}}

=====================
GROUP MEMBERS
=====================
{members_json}

Return JSON array only.
"""




#endregion#? Entity Resolution Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Class Recognition Prompt



# -------------------------
# Class Recognition Prompt (REVISED)
# -------------------------

CLASS_REC_PROMPT_TEMPLATE = """
You are a schema / ontology class suggester for a DOMAIN-AGNOSTIC knowledge graph generator.

GOAL (THIS STEP ONLY)
Propose candidate ontology CLASSES that group the provided resolved entities into meaningful, reusable schema concepts.
Your output here is an initial suggestion. A later step will refine these classes and assign broader Class_Groups.

CRUCIAL PIPELINE CONTEXT
- The entities you see are ALREADY resolved/canonicalized upstream.
- You MUST NOT rename, edit, or reinterpret entity identity here.
- Your only job is to propose class candidates (class_label + members).
- A later "class resolution" step will:
  * merge/split classes,
  * reassign entities across classes,
  * and assign a BROADER Class_Group above classes.
So: focus on producing high-quality, reusable class_label values (the main ontology class level).

INPUT YOU ARE GIVEN (per entity)
- entity_id: unique id (use this in member_ids)
- entity_name: canonical entity name
- entity_description: concise meaning
- resolution_context: 20–120 word excerpt; PRIMARY evidence for meaning
- entity_type_hint: weak hint; may be noisy or wrong
- node_properties: optional intrinsic properties

YOUR TASK
Given a small group of entities, return ZERO or more class proposals.
Each proposed class should group entities that truly share a coherent ontological type.

CLASS LABEL QUALITY (MOST IMPORTANT)
A good class_label should look like something a human ontology expert would write:
- Reusable beyond this specific document.
- Readable, stable, and meaningful in the real world.
- Short noun phrase (1–3 words). Prefer singular (e.g., "City", "Novel", "Payment method").
- Avoid ultra-broad labels (e.g., "Thing", "General", "Misc") unless forced in single-entity mode.
- Avoid instance-specific labels (e.g., "Pump P-101", "Chapter 2 topics", "2020 failures").
- Choose abstraction level based on the member set:
  * If members are sibling INSTANCES → label their shared type (Paris+Berlin+Tokyo → "City").
  * If members are sibling SUBTYPES → label their shared parent type (Western+Documentary+Sci‑fi → "Film genre").
  * Do NOT collapse unrelated entities using vague umbrellas.

CLASS TYPE HINT (SECONDARY; OPEN-WORLD)
- class_type_hint is a SHORT, broad family/role hint for the class.
- This is NOT a fixed list. Use whatever broad hint best fits.
- Keep it concise (1–2 words), reusable, and not instance-specific.
Illustrative examples only: Location, Person, Organization, CreativeWork, Device, Material, Process, Event, Method, Condition, Measurement, Topic.

MEMBERSHIP RULES (ENFORCED BY THE SYSTEM)
- If the input contains MULTIPLE entities: every proposed class MUST have at least TWO member_ids.
- Single-member classes are allowed ONLY when the input contains EXACTLY ONE entity.

SINGLE-ENTITY MODE
If you receive exactly ONE entity:
- Return exactly ONE class containing that entity.
- Use the best reusable class_label you can justify.
- If truly uncertain, output "Uncategorized: <entity_name>" with low confidence (<= 0.20).

OUTPUT FORMAT (REQUIRED)
Return ONLY a JSON ARRAY (no markdown, no commentary).
Each element must have EXACTLY these keys:

{{
  "class_label": (string),
  "class_description": (string, 1–2 sentences; membership criteria + distinction),
  "class_type_hint": (string; short broad hint, or empty string),
  "member_ids": (array[string]; must be entity_ids from the input),
  "confidence": (float 0.0–1.0),
  "evidence_excerpt": (string; optional but recommended, 5–30 words)
}}

NOTES
- Use ONLY provided entity_ids.
- Prefer non-overlapping classes; overlap only if genuinely justified.
- It is OK to return [] when nothing forms a clean reusable class.

EXAMPLES (diverse domains; follow JSON shape exactly)

Example 1 (places → mid-level class)
Input entities:
- Paris (En_1)
- Berlin (En_2)
- Tokyo (En_3)

Output:
[
  {{
    "class_label": "City",
    "class_description": "Urban settlements that function as major population and administrative centers.",
    "class_type_hint": "Location",
    "member_ids": ["En_1","En_2","En_3"],
    "confidence": 0.93,
    "evidence_excerpt": "capital city ... metropolitan area"
  }}
]

Example 2 (arts/media → class depends on member granularity)
Input entities:
- western (En_7)
- documentary (En_8)
- science fiction (En_9)

Output:
[
  {{
    "class_label": "Film genre",
    "class_description": "Genre categories used to classify films by narrative and stylistic conventions.",
    "class_type_hint": "CreativeWork",
    "member_ids": ["En_7","En_8","En_9"],
    "confidence": 0.86,
    "evidence_excerpt": "genre ... documentary ... western"
  }}
]

Example 3 (technical/engineering concepts → reusable process-level class)
Input entities:
- graphitization (En_11)
- sulfidation (En_12)

Output:
[
  {{
    "class_label": "Degradation mechanism",
    "class_description": "Processes that deteriorate material integrity under operating or environmental conditions.",
    "class_type_hint": "Process",
    "member_ids": ["En_11","En_12"],
    "confidence": 0.84,
    "evidence_excerpt": "degradation mechanism ... elevated temperature"
  }}
]

ENTITIES (one per line)
entity_id | entity_name | entity_description | resolution_context | entity_type_hint | node_properties_json
{members_block}

Return JSON array only.
"""




#endregion#? Class Recognition Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Class Resolution Prompt



# -------------------------
# Class Resolution Prompt (REVISED)
# -------------------------

CLASS_RES_PROMPT_TEMPLATE = """
You are an expert schema editor for a DOMAIN-AGNOSTIC knowledge graph generator.

GOAL (THIS STEP ONLY)
You are given a *suggestive* cluster of candidate classes. Produce an ORDERED JSON ARRAY of schema-edit function calls that improves:
1) class_label (the main ontology class level),
2) class_group (a broad connector above classes),
3) member assignments (which entities belong to which classes).

This step refines the schema. Entities are fixed; you are editing the class layer above them.

ABOUT THE INPUT CLUSTER (CRITICAL)
- The cluster was produced automatically from embeddings and is only suggestive.
- It may contain multiple distinct topics.
- Do NOT force all classes into the same class_group or merge them just because they appear together.
- Your job is to correct the cluster into a coherent schema.

TARGET SCHEMA (WHAT "GOOD" LOOKS LIKE)
We want a clean expert-like hierarchy:

Class_Group  →  Class (class_label)  →  Entity

- class_label is the MAIN, mid-level ontology class that does most of the semantic work.
- class_group is a BROADER, more stable parent category that CONNECTS multiple related classes.
- class_type_hint is a weak local hint and may be inconsistent; you may harmonize it, but class_group is the real upper connector.

ABSTRACTION BALANCE (class_label vs class_group)
Choose class_label and class_group jointly so the hierarchy is coherent:
- class_group should be BROADER than the classes it connects, but still meaningful and reusable.
- The “right” breadth depends on what the text/dataset covers:
  * If classes include Film / Music / Novel → class_group could be "Creative work" (broad parent).
  * If classes include Documentary film / Animated film / Feature film → class_group could be "Film" (narrower, but still a parent over those classes).
- Prefer class_group names that help a human navigate the schema and connect sibling classes.
- Avoid "Misc", "General", "Other" unless truly unavoidable; if you use them, add a remark explaining why.

WHEN TO PROPOSE STRUCTURAL CHANGES
Be proactive, but do not invent structure without evidence:
- merge_classes: when two+ classes are the SAME concept (synonyms, pluralization, redundant split).
- split_class: when one class bundles multiple distinct concepts that should become separate classes.
- reassign_entities: when entities are clearly mis-assigned to the wrong class.
- create_class: when a coherent group of entities does not fit any existing class.
- modify_class: for renaming, clarifying descriptions, harmonizing type hints, assigning/fixing class_group, or adding remarks.

It is acceptable that some clusters need only class_group assignment + small metadata fixes.
Return [] only if you are extremely confident that nothing needs improvement (rare).

========================
AVAILABLE FUNCTIONS
========================
Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class", "split_class"]
- "args": arguments as defined below.

========================
ID HANDLING RULES
========================
- You MUST NOT invent real class IDs.
- Use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except for newly merged/created/split classes.
- When you refer to a newly merged/created/split class later in the same output,
  assign a provisional_id (any consistent string) and reuse it consistently.
- After merging, do NOT treat old class_ids as separate; refer to the merged class via provisional_id.

========================
JUSTIFICATION REQUIREMENT
========================
Every function call MUST include:
  "justification": "<one-line reason>"
You MAY include:
  "confidence": <0.0–1.0>
  "remark": <string or null>

========================
EXAMPLES (showing abstraction balance + class_group)
========================

Example A (sibling classes share a broader class_group)
If you see classes like Film / Music / Novel (distinct but related):
- Keep them as separate classes.
- Assign the SAME class_group like "Creative work" to connect them.

Example B (subclasses share a narrower class_group)
If you see classes like Documentary film / Animated film / Feature film:
- Keep them separate classes.
- Assign class_group "Film".

Example C (duplicate classes → merge, then set class_group)
If you see "City" and "Cities" with the same meaning:
- merge_classes to one canonical "City"
- then modify_class to set class_group "Location"

========================
FUNCTION DEFINITIONS
========================

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number between 0 and 1, optional>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>],          # must be from provided entities
  "provisional_id": <string or null>,
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number between 0 and 1, optional>
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>,
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number between 0 and 1, optional>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>,
  "remark": <string or null>,
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

5) split_class
args = {
  "source_class_id": <existing_class_id or provisional_id>,
  "splits": [
    {
      "name": <string or null>,
      "description": <string or null>,
      "class_type_hint": <string or null>,
      "member_ids": [<entity_ids>],      # must be from source_class member_ids
      "provisional_id": <string or null>
    }
  ],
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number between 0 and 1, optional>
}

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
- remarks (optional list)

{cluster_block}

========================
OUTPUT
========================
Return ONLY the JSON array of ordered function calls.
"""


#endregion#? Class Resolution Prompt
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Relation Recognition Prompt

#-------------------------
# Relation Recognition Prompt
#-------------------------

REL_REC_PROMPT_TEMPLATE = """
GOAL: Build a context-enriched knowledge graph (KG) from document text.

THIS STEP (Relation Recognition / Relation Extraction):
Extract directed relation instances between the PROVIDED entities using ONLY the provided chunk text.
Also capture rich relation-level qualifiers (conditions, constraints, modality, etc.).
This step is the last step that can directly see the raw chunk text, so do not drop important context.

=====================
INPUT YOU ARE GIVEN
=====================
You will receive a JSON payload (appended after this prompt) containing:
- chunk_id (string)
- chunk_text (string): the raw text you must extract from
- entities (array): resolved entities that are guaranteed to be conceptually present in this chunk
  Each entity includes: entity_id, entity_name, entity_description, class_label, class_group

IMPORTANT: TRUST THE ENTITIES.
- The exact surface string might not appear verbatim in chunk_text due to normalization, aliases, pronouns, or implicit mentions.
- Do NOT exclude a relation solely because the exact entity_name string is not present.
- However, EVERY relation you output must be supported by chunk_text meaningfully.

=====================
YOUR TASK
=====================
Identify ZERO OR MORE directed relations between the provided entities.

A relation instance connects:
- subject_entity_id  (HEAD / source)
- object_entity_id   (TAIL / target)

Direction rule:
- Choose the direction that best matches how the text expresses “who/what relates to whom/what”.
- If direction is ambiguous or could be inverted, choose a reasonable direction but explain ambiguity in resolution_context or remark.

High recall:
- Prefer recall over precision: if a relation is plausible and text-supported, include it with lower confidence.
- Do NOT invent relations that are mere co-occurrence with no semantic link.

NO SELF-LOOPS (HARD RULE):
- You MUST NOT output relations where subject_entity_id == object_entity_id.
- If the text only conveys an intrinsic fact about one entity, do NOT force an edge.
  Instead, write in remark: "intrinsic_property_candidate: <key>=<value>" (or similar) and omit the relation.

=====================
RELATION INSTANCE FIELDS
=====================
1) relation_name (normalized, reusable)
- Use a short predicate-style label, preferably snake_case.
- It should be reusable across many documents (not tied to this one situation).
- Do NOT include qualifiers/modality/conditions in relation_name.
  Put those in qualifiers / remark instead.

Examples (illustrative only): "causes", "prevents", "requires", "depends_on", "uses", "produces",
"located_in", "part_of", "has_part", "authored", "member_of", "reports", "defines", "compares_to".

2) relation_surface (verbatim evidence phrase)
- A minimal verb phrase or short span from chunk_text that expresses the relation.

3) rel_desc (instance-level)
- Brief explanation of this relation instance (may mention the specific entities and local context).

4) rel_hint_type (MANDATORY broad relation-group token)
For EVERY relation, set rel_hint_type to EXACTLY ONE SINGLE-WORD value from:
IDENTITY, COMPOSITION, CAUSALITY, TEMPORALITY, SPATIALITY, ROLE, PURPOSE, DEPENDENCY,
COUPLING, TRANSFORMATION, COMPARISON, INFORMATION, ASSOCIATION

Notes:
- Choose the most fitting group; put nuance in rel_desc/qualifiers/remark.
- ASSOCIATION is last resort only:
  If you use ASSOCIATION, set confidence <= 0.4 and explain why no other group fits in remark.

Helpful tie-breaks:
- Requirement/constraint/precondition (“requires”, “must”, “depends on”) -> DEPENDENCY
- Intended function/use (“used for”, “designed for”) -> PURPOSE
- Physical/functional interface/connection/flow -> COUPLING
- Change of state/version/replacement/conversion -> TRANSFORMATION
- Reporting/measurement/documentation/evidence -> INFORMATION

=====================
QUALIFIERS (RELATION-LEVEL CONTEXT)
=====================
For each relation, output:

"qualifiers": {
  "TemporalQualifier": string|null,
  "SpatialQualifier": string|null,
  "OperationalConstraint": string|null,
  "ConditionExpression": string|null,
  "UncertaintyQualifier": string|null,
  "CausalHint": string|null,
  "LogicalMarker": string|null,
  "OtherQualifier": string|null
}

Guidance (domain-agnostic):
- TemporalQualifier: time anchoring (e.g., "in 2020", "during the concert", "after installation")
- SpatialQualifier: location/region/where (e.g., "in Phoenix", "on the north side", "within the chamber")
- OperationalConstraint: operating/env constraints (e.g., "at high temperature", "under low light", "with limited bandwidth")
- ConditionExpression: explicit conditional logic (e.g., "if X > 10", "when Y is absent")
- UncertaintyQualifier: modality/hedging (e.g., "may", "likely", "suggests")
- CausalHint: causal cue words beyond the main predicate (e.g., "due to", "because of")
- LogicalMarker: discourse logic (e.g., "if", "unless", "only when")
- OtherQualifier: anything else important; encode as "Type: value" (e.g., "Audience: adults")

Use JSON null (not the string "null") when absent.

=====================
DEDUP WITHIN THIS CHUNK (IMPORTANT)
=====================
Within a single chunk output:
- Do NOT output duplicate relations with the same (subject_entity_id, object_entity_id, relation_name)
  unless the qualifiers are materially different or conflicting.
- If the same fact is stated multiple times, output ONE relation and mention "repeated mention" in remark.

=====================
OUTPUT FORMAT (REQUIRED)
=====================
Return ONLY valid JSON (no markdown, no commentary).

Top-level MUST be exactly:
{ "relations": [ ... ] }

Each relation object MUST include ALL keys below (use null where applicable):
{
  "subject_entity_id": "En_...",
  "object_entity_id": "En_...",
  "relation_surface": "string",
  "relation_name": "string",
  "rel_desc": "string",
  "rel_hint_type": "string",
  "confidence": 0.0,
  "resolution_context": "string or null",
  "justification": "string or null",
  "remark": "string or null",
  "qualifiers": {
    "TemporalQualifier": "string or null",
    "SpatialQualifier": "string or null",
    "OperationalConstraint": "string or null",
    "ConditionExpression": "string or null",
    "UncertaintyQualifier": "string or null",
    "CausalHint": "string or null",
    "LogicalMarker": "string or null",
    "OtherQualifier": "string or null"
  },
  "evidence_excerpt": "short excerpt from chunk_text (<= 40 words)"
}

If no relations exist, return exactly:
{ "relations": [] }

=====================
MINI EXAMPLES (SHAPE + BEHAVIOR; NOT EXHAUSTIVE)
=====================

Example A (qualifier vs predicate):
Text: "Battery overheating can cause device shutdown during peak usage."
-> relation_name: "causes"
-> UncertaintyQualifier: "can"
-> TemporalQualifier or OtherQualifier: "during peak usage"

Example B (spatial + temporal):
Text: "The concert took place in Phoenix on January 20."
-> relation_name: "occurred_in"
-> rel_hint_type: "SPATIALITY"
-> SpatialQualifier: "Phoenix"
-> TemporalQualifier: "on January 20"

Now read the appended JSON payload and return the required JSON output.
"""

#endregion#? Relation Recognition Prompt
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Relation Resolution Prompt

#-------------------------
# Relation Resolution Prompt
#-------------------------

REL_RES_PROMPT_TEMPLATE = """
You are a proactive RELATION-SCHEMA NORMALIZER for a context-enriched KG.

GOAL (THIS STEP):
Given a suggestive cluster of relation INSTANCES, produce an ordered list of schema edits that:
1) Assign / normalize canonical_rel_name + canonical_rel_desc (predicate-level, direction-sensitive)
2) Assign / normalize rel_cls (relation class = reusable family container)
3) Assign / normalize rel_cls_group (very broad semantic group; single token)
4) Merge ONLY exact-duplicate edges when justified (no semantic deletions)

This step is repeated across multiple runs; if values are already filled, you MUST improve them when inconsistent or weak.

=====================
WHAT YOU ARE GIVEN
=====================
A JSON array of relation instances. Each instance includes (not exhaustive):
- relation_id
- relation_name (raw; noisy; from earlier extraction)
- rel_desc (instance-level; noisy)
- rel_hint_type (broad hint; may be noisy)
- canonical_rel_name (often "TBD" initially)
- canonical_rel_desc (often empty)
- rel_cls (often "TBD")
- rel_cls_group (often "TBD")
- subject_entity_id, object_entity_id (FIXED)
- subject/object names + class metadata
- qualifiers (temporal/spatial/etc.)
- confidence, remarks, evidence_excerpt

IMPORTANT:
- The cluster is ONLY suggestive (embedding-based). It can contain multiple different predicates and classes.
- Do NOT force everything into one canonical_rel_name / one rel_cls / one rel_cls_group.
- Think relation-by-relation.

=====================
RELATION SCHEMA LAYERS (CRITICAL DISTINCTION)
=====================
A) canonical_rel_name (predicate label used on KG edges)
- Fine-grained, direction-sensitive, reusable, preferably snake_case.
- Represents the core meaning WITHOUT qualifiers/modality.
- Examples (illustrative): "located_in", "has_part", "part_of", "causes", "prevents",
"depends_on", "used_for", "authored", "reports", "defines", "compares_to".

B) rel_cls (relation class / family container)
- Broader than a single predicate; groups similar canonical_rel_name values.
- Must be meaningful and reusable in an open-world sense (like a human-designed ontology family).
- Avoid: rel_cls that just repeats rel_cls_group (e.g., "composition_relation" when group is COMPOSITION),
  unless you truly cannot name a more specific family.
- Example families: "location_relation", "mereology_relation", "causation_relation",
"authorship_relation", "measurement_reporting_relation", "dependency_constraint_relation".

C) rel_cls_group (very broad group; SINGLE WORD; FIXED VOCAB)
For THIS pipeline, rel_cls_group MUST be exactly ONE of:
IDENTITY, COMPOSITION, CAUSALITY, TEMPORALITY, SPATIALITY, ROLE, PURPOSE, DEPENDENCY,
COUPLING, TRANSFORMATION, COMPARISON, INFORMATION, ASSOCIATION

- ASSOCIATION is last resort only; if used, explain why and prefer lower confidence.

Strong default:
- If rel_hint_type is correct, prefer rel_cls_group = rel_hint_type.
- If rel_hint_type is wrong, correct it by setting rel_cls_group appropriately.

=====================
WHAT YOU MUST / MUST NOT DO
=====================
MUST:
- Normalize awkward / overly-specific raw relation_name variants into stable canonical_rel_name values.
- Harmonize rel_cls and rel_cls_group for relations that truly share semantics.
- Propose corrections in refinement runs (do not be passive).

MUST NOT:
- Change subject_entity_id or object_entity_id (entities are fixed).
- Delete relation instances for “being noisy”.
- Merge relations that are merely similar but not exact duplicates.

=====================
EXACT DUPLICATE EDGE MERGING (MANDATORY WHEN APPLICABLE)
=====================
You MUST merge relations ONLY when ALL are true:
1) subject_entity_id is identical
2) object_entity_id is identical
3) canonical_rel_name is identical (same meaning + same direction after your normalization)

Qualifier rule:
- If qualifiers match or one is a strict subset of the other: merge and keep the more informative qualifiers.
- If qualifiers conflict materially: DO NOT merge; keep both and add a remark explaining the conflict.

IMPORTANT PRACTICAL RULE:
- If two relations are duplicates except that schema fields differ (e.g., rel_cls differs),
  you should first normalize schema (modify_rel_schema / set_rel_cls / set_rel_cls_group),
  then merge_relations into one consistent edge.

=====================
OUTPUT: FUNCTION-CALL LIST ONLY
=====================
Return ONLY a JSON ARRAY of ordered function calls.


Allowed functions ONLY:
1) set_canonical_rel
2) set_rel_cls
3) set_rel_cls_group
4) modify_rel_schema
5) add_rel_remark
6) merge_relations

Order matters (later steps may rely on earlier ones).

-----------------------
Function definitions
-----------------------

1) set_canonical_rel
args = {
  "relation_ids": [<relation_id>...],
  "canonical_rel_name": <string>,
  "canonical_rel_desc": <string or null>,
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number 0..1, optional>
}

2) set_rel_cls
args = {
  "relation_ids": [<relation_id>...],
  "rel_cls": <string>,
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number 0..1, optional>
}

3) set_rel_cls_group
args = {
  "relation_ids": [<relation_id>...],
  "rel_cls_group": <string>,   # MUST be one of the allowed single-word groups
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number 0..1, optional>
}

4) modify_rel_schema
Use to correct any combination of canonical_rel_name / canonical_rel_desc / rel_cls / rel_cls_group.
args = {
  "relation_ids": [<relation_id>...],
  "canonical_rel_name": <string or null>,
  "canonical_rel_desc": <string or null>,
  "rel_cls": <string or null>,
  "rel_cls_group": <string or null>,  # MUST be one of the allowed single-word groups if not null
  "justification": <string>,
  "remark": <string or null>,
  "confidence": <number 0..1, optional>
}

5) add_rel_remark
args = {
  "relation_ids": [<relation_id>...],
  "remark": <string>,
  "justification": <string>,
  "confidence": <number 0..1, optional>
}

6) merge_relations
Use ONLY for exact duplicates.
args = {
  "relation_ids": [<relation_id>...],             # at least 2
  "provisional_id": "MERGE(RelR_a|RelR_b|...)",
  "subject_entity_id": <string>,
  "object_entity_id": <string>,
  "canonical_rel_name": <string>,
  "canonical_rel_desc": <string or null>,
  "new_rel_cls": <string or null>,
  "new_rel_cls_group": <string or null>,          # allowed single-word groups only
  "relation_name": <string or null>,              # optional traceability
  "rel_desc": <string or null>,                   # optional
  "rel_hint_type": <string or null>,              # optional
  "subject_entity_name": <string or null>,
  "object_entity_name": <string or null>,
  "qualifiers": <dict or null>,                   # merged qualifiers
  "remark": <string or null>,
  "justification": <string>,
  "confidence": <number 0..1, optional>
}

=====================
MINI EXAMPLES (BETTER, DOMAIN-AGNOSTIC)
=====================

Example 1 (normalize variants):
- RelR_1: relation_name="is located in"
- RelR_2: relation_name="situated in"
Same direction (Place -> Region)

Actions (illustrative):
[
  {
    "function": "set_canonical_rel",
    "args": {
      "relation_ids": ["RelR_1","RelR_2"],
      "canonical_rel_name": "located_in",
      "canonical_rel_desc": "Indicates that the subject is geographically located within the object.",
      "justification": "Both edges express the same spatial containment meaning with the same direction.",
      "remark": null,
      "confidence": 0.9
    }
  },
  {
    "function": "set_rel_cls",
    "args": {
      "relation_ids": ["RelR_1","RelR_2"],
      "rel_cls": "location_relation",
      "justification": "Both are geographic placement predicates.",
      "remark": null,
      "confidence": 0.85
    }
  },
  {
    "function": "set_rel_cls_group",
    "args": {
      "relation_ids": ["RelR_1","RelR_2"],
      "rel_cls_group": "SPATIALITY",
      "justification": "This is a spatial relation.",
      "remark": null,
      "confidence": 0.95
    }
  }
]

Example 2 (exact-duplicate merge after normalization):
- RelR_7 and RelR_9 share the same subject_entity_id + object_entity_id + canonical_rel_name
- RelR_9 has richer qualifiers

Actions (illustrative):
[
  {
    "function": "merge_relations",
    "args": {
      "relation_ids": ["RelR_7","RelR_9"],
      "provisional_id": "MERGE(RelR_7|RelR_9)",
      "subject_entity_id": "En_A",
      "object_entity_id": "En_B",
      "canonical_rel_name": "depends_on",
      "canonical_rel_desc": "Indicates that the subject requires the object as a prerequisite or dependency.",
      "new_rel_cls": "dependency_constraint_relation",
      "new_rel_cls_group": "DEPENDENCY",
      "qualifiers": {
        "TemporalQualifier": null,
        "SpatialQualifier": null,
        "OperationalConstraint": "under limited resources",
        "ConditionExpression": null,
        "UncertaintyQualifier": null,
        "CausalHint": null,
        "LogicalMarker": "if",
        "OtherQualifier": null
      },
      "remark": "Merged exact duplicate edges; kept richer qualifiers.",
      "justification": "Same head/tail and same predicate meaning; qualifiers are compatible (one is more informative).",
      "confidence": 0.9
    }
  }
]

Now read the provided relation-instance cluster and return ONLY the JSON array of function calls.

--------------------------------
INPUT RELATIONS (THIS CHUNK)
--------------------------------

Below is the JSON array of relation instances in THIS PART of a cluster
(we split very large clusters into smaller chunks just to keep the prompt size manageable):

{cluster_block}

--------------------------------
OUTPUT
--------------------------------

Return ONLY the JSON ARRAY of function calls, e.g.:

[
  {
    "function": "set_canonical_rel",
    "args": { ... }
  },
  {
    "function": "set_rel_cls",
    "args": { ... }
  },
  ...
]


Now, produce the output exactly as specified. Producing anything other than the JSON array with specified ALLOWED function calls will cause termination of the pipeline.


"""


#endregion#? Relation Resolution Prompt
#?#########################  End  ##########################




#endregion#! Second Version
#!#############################################  End Chapter  ##################################################