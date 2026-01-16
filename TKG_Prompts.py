import json

#?######################### Start ##########################
#region:#?   Entity Recognition Prompt


# -------------------------
# Entity Recognition Prompt
# -------------------------
ENT_REC_SUGGESTED_TYPES = [
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
    "YOU MAY USE ANY OTHER TYPE THAT FITS BETTER",
]

ENT_REC_EXAMPLES = [
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
            {
                "prop_name": "material_grade",
                "prop_value": "304",
                "justification": "explicit explanation of inclusion in the FOCUS chunk"
            }
        ]
    }
]

ENT_REC_EXAMPLES_JSON = json.dumps(ENT_REC_EXAMPLES, ensure_ascii=False, indent=2)

ENT_REC_PROMPT_TEMPLATE = """GOAL: We are creating a context-enriched knowledge graph (KG) from textual documents.
YOUR TASK (THIS STEP ONLY): Extract entity mentions from the FOCUS chunk ONLY. Relation-level qualifiers, conditions, and other contextual information will be extracted later in the RELATION EXTRACTION step (Rel Rec).

PRINCIPLES (read carefully):
- Extract broadly: prefer recall (extract candidate mentions). Do NOT be conservative. When in doubt, include the candidate mention. Later stages will cluster, canonicalize, and resolve.
- Ground every output in the FOCUS chunk. You may CONSULT CONTEXT (previous chunks concatenated) only for disambiguation/pronoun resolution.
- DO NOT output relation-level qualifiers, situational context, or evidential/epistemic markers in this step. The ONLY exception is truly intrinsic node properties (see 'INTRINSIC NODE PROPERTIES' below).
- The suggested type hints below are guidance — you may propose more specific domain-appropriate types.

CORE INSTRUCTION FOR CONCEPTUAL ENTITIES (ENTITY-LEVEL, NOT CLASSES):
- For recurring concepts (phenomena, processes, failure modes, behaviors, conditions, states, methods), extract a SHORT, STABLE, REUSABLE entity-level label as `entity_name`.
- `entity_name` is a canonical mention-level surface form (normalized for this mention). It is NOT an ontology or Schema class label. If you think a class is relevant, place it in `entity_type_hint`.
- If the text describes the concept indirectly (e.g., 'this type of …', 'loss of … under conditions'), infer the best short label (e.g., 'graphitization') and put evidence in `entity_description` and `resolution_context`.
- If unsure of the label, still propose it and lower the `confidence_score` (e.g., 0.5–0.7). We prefer 'extract first, judge later'.

INTRINSIC NODE PROPERTIES:
- You MUST include `node_properties` when the property is identity-defining for the entity (removing it would change what the entity fundamentally is) and they are MANDATORY when present in FOCUS chunk.
- Intrinsic property means ANY stable attributes that define identity no matter what (removing it would change what the entity fundamentally is) e.g material_grade='304', e.g., chemical formula, etc.
- Expectation: IF the property is defined in relation to other entities, that is not intrinsic, therefore we postpone them to Rel Rec.

CONFIDENCE GUIDELINES:
- 0.90 - 1.00 : Certain — explicit mention in FOCUS chunk, clear support.
- 0.70 - 0.89 : Likely — supported by FOCUS or resolved by CONTEXT.
- 0.40 - 0.69 : Possible — plausible inference; partial support.
- 0.00 - 0.39 : Speculative — weakly supported; include only if likely useful.

SUGGESTED TYPE HINTS (Just to give you an idea! You may propose any other type):
- {suggested_types}

OUTPUT FORMAT INSTRUCTIONS (REVISED — REQUIRED):
- Return ONLY a single JSON array (no extra commentary, no markdown fences).
- Each element must be an object with the following keys (exact names):
   * entity_name (string) — short canonical surface label for the mention (mention-level, NOT class).
   * entity_description (string) — 10–25 word description derived from the FOCUS chunk (and CONTEXT if needed).
   * entity_type_hint (string) — suggested type (from list or a better string).
   * context_phrase (string) — short (3–10 word) excerpt from the FOCUS chunk that PROVES the mention provenance (required when possible).
   * resolution_context (string) — minimal 20–120 word excerpt that best explains WHY this mention maps to `entity_name`. Prefer the sentence containing the mention and at most one neighbor sentence; if CONTEXT was required, include up to one supporting sentence from CONTEXT.
   * confidence_score (float) — 0.0–1.0.
   * node_properties (array of objects, Include ONLY if an intrinsic property is present in the focus chunk) — Each: {{ 'prop_name': str, 'prop_value': str|num, 'justification': str }}.

IMPORTANT:
- DO NOT list entities that appear only in CONTEXT. Only extract mentions present in the FOCUS chunk.
- DO NOT output relation qualifiers, situational context, causal hints, or uncertainty markers here. Postpone them to Rel Rec.
- Do not output ontology-level class names as `entity_name`. If relevant, place such information in `entity_type_hint` and keep `entity_name` a mention-level label.
- For conceptual entities that are described indirectly, prefer a short canonical mention and keep the descriptive evidence in `entity_description` and `resolution_context`.

EMBEDDING WEIGHT NOTE (for clustering later):
WEIGHTS = {{"name": 0.45, "desc": 0.25, "resolution_context": 0.25, "type": 0.05}}
Build resolution_context precisely — it is the second-most important signal after name and description.

=== CONTEXT (previous chunks concatenated for disambiguation) ===
{context_block}

=== FOCUS CHUNK (extract from here) ===
FOCUS_CHUNK_ID: {focus_chunk_id}
{focus_text}

EXAMPLE OUTPUT (three diverse examples — strictly follow JSON shape):
{examples_json}
"""


#endregion#? Entity Recognition Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Entity Resolution Prompt



# -------------------------
# Entity Resolution Prompt
# -------------------------
ENT_RES_PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
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




#endregion#? Entity Resolution Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Class Recognition Prompt


# -------------------------
# Class Recognition Prompt
# -------------------------


CLASS_REC_PROMPT_TEMPLATE = """
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




#endregion#? Class Recognition Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Class Resolution Prompt


# -------------------------
# Class Resolution Prompt
# -------------------------




CLASS_RES_PROMPT_TEMPLATE = """
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
  outside the scope of structural changes (e.g., upstream entity resolution that you suspect is wrong, or entities that
  look identical but should not be changed here), DO NOT try to fix them via merge or reassign.
   Instead, attach a human-facing remark via modify_class (using the 'remark' field)
   so that a human can review it later.

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

#endregion#? Class Resolution Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Relation Recognition Prompt


#-------------------------
# Relation Recognition Prompt
#-------------------------


REL_REC_PROMPT_TEMPLATE = """
You are an expert relation extractor for a CONTEXT-ENRICHED knowledge graph.

High-level setting:
- The entities you see are ALREADY resolved and guaranteed to belong to this chunk.
  They may not always appear with exactly the same surface form in the text,
  but you must **trust** that they are conceptually present in this chunk.
- This is the FINAL stage that has direct access to the raw chunk text.
  After this, we will NOT revisit the chunk for more information.
- Our goal is to capture as MUCH context as possible:
  relations PLUS rich qualifiers (conditions, constraints, uncertainty, etc.).

Your inputs:
- A single text chunk from a technical document.
- A list of resolved entities that appear in that chunk.
- Each entity has: entity_id, entity_name, entity_description, class_label, class_group.

Your main task:
- Identify ZERO or MORE **directed** relations between the entities in this chunk.
- A relation is a meaningful, text-supported connection between a HEAD (subject) entity
  and a TAIL (object) entity.
- The graph is DIRECTED. You must choose subject_entity_id and object_entity_id
  based on how the text expresses the relationship.
  Note: Some relations may be conceptually symmetric or bidirectional; in such cases,
  choose a reasonable subject → object direction for representation purposes,
  and explain any ambiguity in justification or remark.


Very important principles:

1) TRUST THE ENTITIES (we performed entity resolution)
   - We have already run entity recognition & resolution upstream; the provided entities are canonical / resolved mentions derived from the document.
   - Do NOT question whether an entity belongs to this chunk. The exact surface string may not appear verbatim in the chunk because:
       * the entity was canonicalized (normalized) during entity resolution,
       * the chunk uses synonyms, abbreviations, pronouns, or implicit references,
       * the entity was merged from multiple mentions across the section.
   - Use the provided entity_name, entity_description,
     class_label, and class_group to recognize mentions and evidence even when the exact surface form is absent.
   - You may create relations between any pair of provided entities if the chunk text supports the relation conceptually — prefer recall over rejecting a plausible relation solely because the surface token doesn't match exactly.

2) HIGH RECALL & COVERAGE (discover first, refine later)
   - Assume that almost all informational content in the chunk should end up in the KG
     either as a relation or as qualifiers on relations.
   - Do NOT limit discovery based on naming concerns.
   - It is BETTER to propose a relation with lower confidence (and explain your doubts)
     than to miss a real relation or important qualifier.
   - If you are unsure, still output the relation with lower `confidence` and explain
     your uncertainty in `justification` and/or `remark`.

3) QUALIFIERS LIVE ON RELATIONS
   - Intrinsic node properties were (mostly) already captured during entity recognition/resolution.
   - Here, we treat almost all remaining contextual information as relation-level qualifiers.
   - Do NOT try to create or modify entity properties.
   - If you spot something that looks like a missing intrinsic property,
     mention it in `remark` instead of modeling it as a property.

4) CAPTURE QUALIFIERS EVEN IF THE RELATION IS UNCERTAIN
   - If you clearly see a contextual piece (condition, constraint, etc.) that SHOULD be attached
     to some relation but you struggle to identify the exact relation semantics:
       * Make a best-effort guess for the relation_name between the most relevant entities.
       * Set a lower `confidence`.
       * In `remark`, clearly state that this relation was primarily created to capture that qualifier
         and describe the issue (e.g., "relation semantics unclear", "heuristic relation choice").
   - This ensures we do not lose important context even when relation semantics are fuzzy.

5) DO NOT BE OVERLY BIASED BY EXAMPLES
   - Any examples of relation names, relation types, or qualifier values in this prompt are
     **illustrative, not exhaustive**.
   - You are free to discover ANY relation that is supported by the text.
   - Guidance below affects how relations should be *expressed*, not which relations you should find.


-------------------------------------------
Relation naming (VERY IMPORTANT):
-------------------------------------------
- relation_name:
    - A short, normalized phrase describing the CORE semantic relation between subject and object.
    - The goal is NOT to restrict which relations you discover,
      but to express discovered relations in a reusable, abstract form.
    - relation_name SHOULD be something that could reasonably apply to many different
      entity pairs across the corpus, even if the surface wording differs.
    - If the relation meaning is specific to this instance,
      capture that specificity in rel_desc, qualifiers, or remark — not in relation_name.
    - Examples (NOT exhaustive): "causes", "occurs_in", "is_part_of", "used_for",
      "located_in", "prevents", "requires", "correlates_with", "associated_with".
    - Do NOT include qualifiers like "at high temperature", "during startup", "may"
      in relation_name.

- relation_surface:
    - The exact phrase or minimal text span from the chunk that expresses the relation.
    - This MAY be instance-specific and does NOT need to be reusable.
    - Examples (NOT exhaustive): "leads to", "results in", "is part of",
      "used for controlling", "associated with".


-------------------------------------------
Relation hint type (rel_hint_type):
-------------------------------------------
- A short free-form hint describing the nature of the relation
  (e.g., causal, dependency, containment, usage, constraint, correlation, requirement, etc.).
- This is ONLY a hint to help later relation resolution and grouping.
- There is NO fixed list; choose whatever best fits the relation.
- If unsure, still provide your best guess and explain uncertainty in justification.
- It must be a short phrase (1–3 words), not a sentence.


-------------------------------------------
Qualifiers:
-------------------------------------------
For each relation, fill this dict (values are strings or null):

  "qualifiers": {
    "TemporalQualifier": "... or null",
    "SpatialQualifier": "... or null",
    "OperationalConstraint": "... or null",
    "ConditionExpression": "... or null",
    "UncertaintyQualifier": "... or null",
    "CausalHint": "... or null",
    "LogicalMarker": "... or null",
    "OtherQualifier": "expectedType: value or null"
  }

Meanings (examples are illustrative, NOT exhaustive):
- TemporalQualifier:
    when something holds (e.g., "during heating", "after 1000h", "at 25°C").
- SpatialQualifier:
    where something holds (e.g., "near weld", "in heat-affected zone").
- OperationalConstraint:
    operating or environmental conditions (e.g., "elevated temperature", "high load").
- ConditionExpression:
    explicit conditional or threshold clauses (e.g., "temperature > 450°C").
- UncertaintyQualifier:
    modality or hedging (e.g., "may", "likely", "suspected").
- CausalHint:
    lexical causal cues beyond the main verb (e.g., "due to", "caused by").
- LogicalMarker:
    discourse or logic markers (e.g., "if", "when", "unless").
- OtherQualifier:
    use when a qualifier does not fit the above categories
    (encode both expected type and value, e.g., "MeasurementContext: laboratory test").

If there are multiple "other" qualifiers, you may combine them in a single string.


-------------------------------------------
Other fields (and how to use them):
-------------------------------------------
- confidence:
    - float between 0 and 1 (your estimated confidence that this relation is correctly captured).
    - When in doubt, still output the relation but with a lower confidence (e.g., 0.2–0.4).

- rel_desc:
    - A brief instance-level explanation of how the subject and object are related here.
    - This is evidence-oriented and may mention the specific entities involved.

- resolution_context:
    - Short text intended to help later relation resolution / canonicalization
      (e.g., why a certain direction was chosen, or how this phrasing compares to others).

- justification:
    - Explanation of how the chunk text supports this relation.
    - Also state doubts here if semantics are ambiguous.

- remark:
    - Free-text notes for meta-issues or edge cases:
        * "relation created mainly to capture qualifier X"
        * "may reflect document organization rather than domain semantics"
        * "possible intrinsic property not captured upstream"


-------------------------------------------
Output format:
-------------------------------------------
- You MUST output a single JSON object with exactly one top-level key "relations":
  { "relations": [ ... ] }

- Each relation object MUST have this exact shape (all keys present, use null if not applicable):

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
    "evidence_excerpt": "short excerpt from the chunk text (<= 40 words)"
  }

- subject_entity_id and object_entity_id MUST be chosen from the provided entities.
- You may propose relations between ANY pair of provided entities if the chunk supports it.
- If there are no relations at all, return: { "relations": [] }
- rel_hint_type must be a short phrase (1–3 words), not a sentence.

Coverage:
- Try to cover ALL meaningful connections and contextual information that can be attached
  to those connections, even if it means producing low-confidence relations with detailed remarks.

Return ONLY valid JSON. No extra commentary.
"""



#endregion#? Relation Recognition Prompt
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Relation Resolution Prompt


#-------------------------
# Relation Resolution Prompt
#-------------------------

REL_RES_PROMPT_TEMPLATE = """
You are a very proactive RELATION-RESOLUTION assistant, and an expert in knowledge graphs (KGs) and schema (ontology) design.

You are given a CLUSTER of relation INSTANCES from a context-enriched technical KG.
Each relation instance already connects specific subject and object entities, and has:

- relation_id
- relation_name        (raw, from Relation Recognition; may be noisy)
- rel_desc             (instance-level description; may be verbose)
- rel_hint_type        (short free-form hint; may be noisy)
- canonical_rel_name   (often "TBD" initially)
- canonical_rel_desc   (often empty initially)
- rel_cls              (often "TBD" initially)
- rel_cls_group        (often "TBD" initially)
- subject_entity_name, object_entity_name
- subject_class_label, subject_class_group
- object_class_label, object_class_group
- qualifiers (temporal, spatial, etc.)
- confidence, remarks, etc.

IMPORTANT CONTEXT ABOUT THE PIPELINE
------------------------------------
- ALL these fields (especially relation_name, rel_desc, and rel_hint_type) were
  generated by previous LLM stages. They are **approximate** and may be:
    - awkward in wording,
    - too specific,
    - too generic,
    - or slightly wrong.
- This step follows a "generate first, refine later" philosophy.
  Your job is to **refine, disambiguate, normalize, and improve human readability as a KG expert**, not to copy the wording blindly.
- Use upstream fields (especially relation_name, rel_desc, and rel_hint_type) as **semantic hints**, not as final label candidates AT ALL!

MULTI-RUN / REFINEMENT INSTRUCTIONS (READ ONLY WHEN THE FOLLOWING FIELDS ARE ALREADY FILLED: canonical_rel_name, canonical_rel_desc, rel_cls, rel_cls_group)
- Only read/apply the following guidance when at least one of the fields above is NOT "TBD" for relations in this chunk. It means we are in a refinement run.
- Do NOT be passive: if any pre-filled value is inconsistent, ambiguous, or improvable, you MUST propose a correction (use modify_rel_schema) with a concise justification.
- Only return [] if there is extremely strong evidence that no change is required (this is rare).
- This is an iterative process — act now with well-justified corrections using modify_rel_schema (include a short justification), rather than deferring small but meaningful fixes.
- This step will be repeated in later iterations; reasonable but imperfect changes can be corrected later. It is worse to miss a necessary change than to propose a well-justified change that might be slightly adjusted later.


ABOUT CLUSTERS (CRITICAL)
-------------------------
- The cluster you see is produced automatically from embeddings.
- The cluster is **only suggestive**, NOT a hard class:
    - It may contain multiple different canonical relations.
    - It may contain multiple different relation classes.
    - It may contain multiple different relation class groups.
- Your task is **NOT**:
    - "Find ONE canonical_rel_name that covers (almost) all relations in the cluster."
    - "Force all relations in the cluster into the same rel_cls or rel_cls_group."
- Your task **IS**:
    - For EACH relation instance, decide what canonical_rel_name, rel_cls,
      and rel_cls_group are appropriate (especially from the lens of a KG and Schema expert).
    - If some instances do NOT naturally share semantics, assign them different
      canonical_rel_name / rel_cls / rel_cls_group, even though they are
      in the same cluster.
    - If some other instances in the cluster naturally share exactly the same
      semantics, group them together in the same function call.

CRUCIAL DISTINCTION (SCOPE)
---------------------------
We use a 3-layer abstraction for relations:

1) canonical_rel_name
   - Fine-grained, predicate-level.
   - Groups relations that use essentially the SAME predicate meaning and direction.
   - This is the predicate that will be used as the edge label in the KG.
   - Examples: "occurs_in", "resists", "provides_resistance_to", "is_subtype_of",
     "has_metallurgical_structure", "grouped_with".

2) rel_cls
   - A broader relation CLASS that may cover multiple canonical_rel_name values.
   - Think of this as a family of similar predicates.
   - Examples (illustrative): "alloying_element_relation", "microstructure_relation",
     "hierarchy_relation", "corrosion_resistance_relation", "document_series_relation".
   - IMPORTANT: rel_cls should be more specific than broad groups, but more general
     than a single canonical_rel_name.

3) rel_cls_group
   - Very broad semantic DIMENSIONS:
       COMPOSITION, CAUSALITY, IDENTITY, TEMPORALITY, SPATIALITY, AGENCY,
       ASSOCIATION, MODIFICATION, USAGE, REQUIREMENT, etc.
   - This list is illustrative, NOT exhaustive.
   - Example:
       canonical_rel_name: "contains_alloying_element"
       rel_cls:            "alloying_element_relation"
       rel_cls_group:      "COMPOSITION"

RELATION-BY-RELATION THINKING (SUPER IMPORTANT)
-----------------------------------------------
For every relation instance in the cluster, conceptually ask:

1) What is the **best canonical predicate** (canonical_rel_name) for THIS instance,
   ignoring awkward wording in relation_name / rel_hint_type?
2) Which broader **relation class** (rel_cls) best describes this connection?
3) Which **rel_cls_group** (COMPOSITION, CAUSALITY, etc.) does it belong to?

Then:
- If you see other instances in the cluster with the **same semantics**, you may
  include them in the same function call (same canonical_rel_name / rel_cls / rel_cls_group).
- Do NOT try to find one label that covers ALL instances in the cluster.

NOISY HINTS AND HOW TO USE THEM
-------------------------------
- relation_name, rel_desc, rel_hint_type are hints, not perfect labels.
- Examples of bad patterns that you should NOT copy directly:
    - rel_hint_type = "co-classification" → this is awkward wording; interpret it
      as "these things are grouped together as co-equal categories" and choose
      a better canonical name / class (e.g., "grouped_with" with rel_cls
      "coequal_category_relation", rel_cls_group "ASSOCIATION" or "IDENTITY").
    - rel_hint_type = "causal" → this suggests rel_cls_group "CAUSALITY",
      but is NOT a good canonical_rel_name or rel_cls by itself.
- Also avoid redundant patterns like:
    - rel_cls = "composition_relation" when rel_cls_group = "COMPOSITION".
      In that case, choose a more specific rel_cls label, e.g.,
      "alloying_element_relation" or "microstructure_relation".

DEDUPLICATION & NORMALIZATION
-----------------------------
- Within a cluster, if you see multiple predicate variants that are semantically
  the same (e.g., "has_subtype", "is_subtype_of"), you MUST normalize them
  to **one** canonical_rel_name and apply it consistently:
    - For hierarchical relations, prefer a single consistent style, e.g., "is_subtype_of".
- Avoid creating trivial variants that only differ in small wording:
    - Do NOT keep both "provides_resistance_to" and "improves_resistance_to"
      if they are used identically for the same semantic link; pick one.
- Avoid rel_cls that simply repeats rel_cls_group with "_relation" added
  (e.g., "composition_relation" when rel_cls_group is "COMPOSITION").

WHAT YOU NEVER CHANGE
----------------------
- SUBJECT and OBJECT entities are fixed; you MUST NOT change them.
- You NEVER delete relation instances.
- You NEVER move qualifiers from relations onto nodes; qualifiers stay on the
  relation instance.

YOUR FUNCTION VOCABULARY
------------------------
You must return an ordered JSON ARRAY of function calls using ONLY:

1) set_canonical_rel
2) set_rel_cls
3) set_rel_cls_group
4) modify_rel_schema
5) add_rel_remark

Think RELATION-BY-RELATION first, and only group multiple relation_ids into the
same function call when you are genuinely sure they share the same semantics.

--------------------------------
FUNCTION DEFINITIONS
--------------------------------

1) set_canonical_rel
   Use this when you want to SET or ALIGN canonical_rel_name / canonical_rel_desc
   for one or more relation instances (especially when values are "TBD").

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "canonical_rel_name": <string>,           # REQUIRED, normalized predicate for KG edge
     "canonical_rel_desc": <string or null>,   # OPTIONAL, reusable description of this predicate
     "justification": <string>,                # REQUIRED: why these instances share this canonical predicate
     "remark": <string or null>,               # optional human-facing notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - canonical_rel_name should be a short predicate-style phrase, e.g. "occurs_in",
     "resists", "provides_resistance_to", "is_subtype_of", "has_metallurgical_structure".
   - Do NOT include qualifiers (e.g., "at high temperature") here.
   - Use this on relatively TIGHT groups of semantically equivalent predicates.
   - It is perfectly fine to call set_canonical_rel with a **single** relation_id
     when others in the cluster do not share the semantics.


2) set_rel_cls
   Use this when you want to SET or ALIGN rel_cls for one or more instances,
   grouping them into a broader relation class (family).

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "rel_cls": <string>,                      # REQUIRED: class name (e.g., "structure_relation")
     "justification": <string>,                # REQUIRED: why these instances belong to this class
     "remark": <string or null>,               # optional notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - A rel_cls usually covers MULTIPLE canonical_rel_name values, not just one.
   - Think of rel_cls as a conceptual family: e.g. "alloying_element_relation",
     "microstructure_relation", "document_series_relation".
   - Do NOT simply repeat rel_cls_group (e.g. avoid "composition_relation" when
     rel_cls_group is "COMPOSITION") unless there is truly no more specific
     and meaningful class you can provide.


3) set_rel_cls_group
   Use this when you want to SET or ALIGN rel_cls_group for one or more instances
   (or their classes), using a very broad semantic category.

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "rel_cls_group": <string>,                # REQUIRED: broad group (e.g., "COMPOSITION")
     "justification": <string>,                # REQUIRED: why this group fits
     "remark": <string or null>,               # optional notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - rel_cls_group is broad and somewhat orthogonal.
   - Typical groups: COMPOSITION, CAUSALITY, IDENTITY, TEMPORALITY, SPATIALITY,
     AGENCY, INTERACTION, ASSOCIATION, MODIFICATION, USAGE, REQUIREMENT, etc.
   - General hints like rel_hint_type="causal" are better mapped to rel_cls_group
     (CAUSALITY) than to canonical_rel_name.


4) modify_rel_schema
   Use this to REFINE or CORRECT existing schema fields for one or more relations
   (canonical_rel_name / canonical_rel_desc / rel_cls / rel_cls_group).

   args = {
     "relation_ids": [<relation_id>...],            # REQUIRED, at least 1
     "canonical_rel_name": <string or null>,        # OPTIONAL: new canonical name
     "canonical_rel_desc": <string or null>,        # OPTIONAL: new canonical description
     "rel_cls": <string or null>,                   # OPTIONAL: new relation class
     "rel_cls_group": <string or null>,             # OPTIONAL: new broad group
     "justification": <string>,                     # REQUIRED: why this modification is needed
     "remark": <string or null>,                    # optional notes / flags
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - Do NOT be passive: if any pre-filled value is inconsistent, ambiguous, or improvable, you MUST propose a correction using modify_rel_schema.
   - This is iterative: make well-justified corrections now (include a short justification); later runs may further refine them. 
   - It is worse to miss a necessary change than to propose a justified change that might be slightly adjusted later.
   - You MUST NOT try to change (poor) relation_names. As long as the canonical_rel_name, rel_cls, and rel_cls_group are fine, we are good. 
     But if they need correction, YOU MUST use modify_rel_schema to correct them. We will not use raw relation name in the KG. 
   - You MAY use this to normalize even minor variants (e.g., consolidate "has_subtype"
   and "is_subtype_of" to "is_subtype_of") when needed. We want consistent schema.
   


5) add_rel_remark
   Use this when you ONLY want to attach a human-facing remark / caveat / TODO
   without changing any schema fields.

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "remark": <string>,                        # REQUIRED: the remark text
     "justification": <string>,                # REQUIRED: why this remark is useful
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - Use this for issues outside scope (e.g., upstream entity resolution suspicion), or
     to flag ambiguous or borderline cases for later human review.

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

- If this chunk is already perfect (rare), you MAY return [].
- Every call MUST include a "justification".
- Use "remark" (or add_rel_remark) to flag issues or uncertainties that are outside
  the scope of these functions.
"""

#endregion#? Relation Resolution Prompt
#?#########################  End  ##########################



















