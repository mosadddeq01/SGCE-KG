



#*######################### Start ##########################
#region:#?   KG Bundle Export


# """
# KG Bundle Export

# Collect everything needed from the whole KG pipeline into ONE file:
# - Classes + entities (from final Class Res overall_summary)
# - Chunks (for evidence and document structure)
# - Relations (from final Rel Res overall_summary, including qualifiers, canonical fields, etc.)
# - Relation schema (canonical_rel_schema, rel_cls_schema, rel_cls_group_schema)

# Output:
#   SGCE-KG/data/KG/kg_bundle.json

# This file is intended as the single input for downstream KG-construction code.
# """

# import json
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional


# # -----------------------
# # Paths / Config
# # -----------------------

# # Final classes + entities after iterative Class Res
# CLS_RES_OVERALL_DIR = Path(
#     "SGCE-KG/data/Classes/Cls_Res_IterativeRuns/overall_summary"
#     )
# CLASSES_AND_ENTITIES_PATH = CLS_RES_OVERALL_DIR / "classes_and_entities.json"

# # Chunks (sentence-level)
# CHUNKS_PATH = Path(
#     "SGCE-KG/data/Chunks/chunks_sentence.jsonl"
# )

# # Final relations + relation schemas after iterative Rel Res
# REL_RES_OVERALL_DIR = Path(
#     "SGCE-KG/data/Relations/Rel Res_IterativeRuns/overall_summary"
# )
# RELATIONS_RESOLVED_JSONL = REL_RES_OVERALL_DIR / "relations_resolved.jsonl"
# RELATIONS_RESOLVED_JSON = REL_RES_OVERALL_DIR / "relations_resolved.json"

# CANONICAL_REL_SCHEMA_PATH = REL_RES_OVERALL_DIR / "canonical_rel_schema.json"
# REL_CLS_SCHEMA_PATH = REL_RES_OVERALL_DIR / "rel_cls_schema.json"
# REL_CLS_GROUP_SCHEMA_PATH = REL_RES_OVERALL_DIR / "rel_cls_group_schema.json"

# # Where to write the final bundle
# KG_BUNDLE_OUT = Path(
#     "SGCE-KG/data/KG/kg_bundle.json"
# )


# # -----------------------
# # Small helpers
# # -----------------------

# def _load_json(path: Path, default: Any) -> Any:
#     if not path.exists():
#         return default
#     try:
#         return json.loads(path.read_text(encoding="utf-8"))
#     except Exception:
#         return default

# def _safe_json_load_line(line: str) -> Optional[Dict]:
#     line = line.strip()
#     if not line:
#         return None
#     try:
#         return json.loads(line)
#     except Exception:
#         return None


# # -----------------------
# # Core export function
# # -----------------------

# def build_kg_bundle() -> None:
#     """
#     Build a single JSON "kg_bundle.json" that contains:

#     {
#       "metadata": {...},
#       "classes": [...],
#       "entities_map": {...},
#       "chunks": { chunk_id: { ...chunk fields... }, ... },
#       "relations": [...],
#       "canonical_rel_schema": [... or {}],
#       "rel_cls_schema": [... or {}],
#       "rel_cls_group_schema": [... or {}]
#     }

#     Relations are taken from the final Rel Res output and are expected to already
#     contain:
#       - relation_id
#       - subject_entity_id / object_entity_id
#       - relation_name (raw)
#       - canonical_rel_name / canonical_rel_desc (if filled by Rel Res)
#       - rel_cls, rel_cls_group (if filled)
#       - qualifiers dict
#       - chunk_id
#       - evidence_excerpt
#       - any other fields kept by the pipeline (rel_desc, rel_hint_type, etc.)
#     """

#     # 1) Load final classes + entities
#     classes_and_entities = _load_json(CLASSES_AND_ENTITIES_PATH, default={"classes": [], "entities_map": {}})
#     classes: List[Dict[str, Any]] = classes_and_entities.get("classes", []) or []
#     entities_map: Dict[str, Dict[str, Any]] = classes_and_entities.get("entities_map", {}) or {}

#     print(f"[KG bundle] Loaded {len(classes)} classes and {len(entities_map)} entities from {CLASSES_AND_ENTITIES_PATH}")

#     # 2) Load chunks
#     chunks_by_id: Dict[str, Dict[str, Any]] = {}
#     if CHUNKS_PATH.exists():
#         with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
#             for line in fh:
#                 obj = _safe_json_load_line(line)
#                 if not obj:
#                     continue
#                 cid = obj.get("id")
#                 if cid:
#                     chunks_by_id[cid] = obj
#         print(f"[KG bundle] Loaded {len(chunks_by_id)} chunks from {CHUNKS_PATH}")
#     else:
#         print(f"[KG bundle] WARNING: chunks file not found at {CHUNKS_PATH}, 'chunks' will be empty.")

#     # 3) Load final relations (prefer jsonl; fallback to json)
#     relations: List[Dict[str, Any]] = []
#     if RELATIONS_RESOLVED_JSONL.exists():
#         with RELATIONS_RESOLVED_JSONL.open("r", encoding="utf-8") as fh:
#             for line in fh:
#                 obj = _safe_json_load_line(line)
#                 if obj:
#                     relations.append(obj)
#         print(f"[KG bundle] Loaded {len(relations)} relations from {RELATIONS_RESOLVED_JSONL}")
#     elif RELATIONS_RESOLVED_JSON.exists():
#         relations = _load_json(RELATIONS_RESOLVED_JSON, default=[])
#         if not isinstance(relations, list):
#             relations = []
#         print(f"[KG bundle] Loaded {len(relations)} relations from {RELATIONS_RESOLVED_JSON}")
#     else:
#         print(f"[KG bundle] WARNING: no final relations file found in {REL_RES_OVERALL_DIR}, 'relations' will be empty.")

#     # 4) Load relation schemas (canonical / class / group)
#     canonical_rel_schema = _load_json(CANONICAL_REL_SCHEMA_PATH, default=[])
#     rel_cls_schema = _load_json(REL_CLS_SCHEMA_PATH, default=[])
#     rel_cls_group_schema = _load_json(REL_CLS_GROUP_SCHEMA_PATH, default=[])

#     print(f"[KG bundle] canonical_rel_schema size: {len(canonical_rel_schema) if isinstance(canonical_rel_schema, list) else 'dict'}")
#     print(f"[KG bundle] rel_cls_schema size: {len(rel_cls_schema) if isinstance(rel_cls_schema, list) else 'dict'}")
#     print(f"[KG bundle] rel_cls_group_schema size: {len(rel_cls_group_schema) if isinstance(rel_cls_group_schema, list) else 'dict'}")

#     # 5) Assemble bundle
#     bundle = {
#         "metadata": {
#             "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#             "source_paths": {
#                 "classes_and_entities": str(CLASSES_AND_ENTITIES_PATH),
#                 "chunks": str(CHUNKS_PATH),
#                 "relations_resolved_jsonl": str(RELATIONS_RESOLVED_JSONL),
#                 "relations_resolved_json": str(RELATIONS_RESOLVED_JSON),
#                 "canonical_rel_schema": str(CANONICAL_REL_SCHEMA_PATH),
#                 "rel_cls_schema": str(REL_CLS_SCHEMA_PATH),
#                 "rel_cls_group_schema": str(REL_CLS_GROUP_SCHEMA_PATH),
#             },
#         },
#         "classes": classes,
#         "entities_map": entities_map,
#         "chunks": chunks_by_id,
#         "relations": relations,
#         "canonical_rel_schema": canonical_rel_schema,
#         "rel_cls_schema": rel_cls_schema,
#         "rel_cls_group_schema": rel_cls_group_schema,
#     }

#     # 6) Write out
#     KG_BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
#     KG_BUNDLE_OUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"[KG bundle] Written KG bundle to {KG_BUNDLE_OUT}")


# # After defining everything, you can call:
# build_kg_bundle()

#endregion#?   KG Bundle Export
#*#########################  End  ##########################






#*######################### Start ##########################
#region:#?   Cypher Queries - V6



MATCH (n)
DETACH DELETE n;





LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL (row) {
  WITH row
  // skip empty rows
  WITH row
  WHERE row.entity_id IS NOT NULL AND row.entity_id <> ''

  MERGE (e:Entity {entity_id: row.entity_id})
  SET
    e.name                 = row.entity_name,
    e.description          = row.entity_description,
    e.type_hint            = row.entity_type_hint,
    e.confidence           = CASE row.entity_confidence
                               WHEN "" THEN NULL
                               ELSE toFloat(row.entity_confidence)
                             END,
    e.resolution_context   = row.entity_resolution_context,
    e.flag                 = row.entity_flag,
    e.class_id             = row.class_id,
    e.class_label          = row.class_label,
    e.class_group          = row.class_group,
    // chunk_ids is a JSON list string
    e.chunk_ids            = apoc.convert.fromJsonList(row.chunk_ids),
    // node_properties is a JSON list string
    e.node_properties      = apoc.convert.fromJsonList(row.node_properties)
} IN TRANSACTIONS OF 1000 ROWS;












LOAD CSV WITH HEADERS FROM 'file:///rels_fixed_no_raw.csv' AS row
CALL (row) {
  WITH row
  // skip rows missing endpoints
  WITH row
  WHERE row.start_id IS NOT NULL AND row.start_id <> ''
    AND row.end_id   IS NOT NULL AND row.end_id   <> ''

  MATCH (s:Entity {entity_id: row.start_id})
  MATCH (o:Entity {entity_id: row.end_id})

  MERGE (s)-[r:RELATION {relation_id: row.relation_id}]->(o)
  WITH r, row, apoc.convert.fromJsonMap(row.qualifiers) AS qmap
  SET
    r.raw_name           = row.raw_relation_name,
    r.canonical_name     = row.canonical_rel_name,
    r.canonical_desc     = row.canonical_rel_desc,
    r.rel_cls            = row.rel_cls,
    r.rel_cls_group      = row.rel_cls_group,
    r.rel_hint_type      = row.rel_hint_type,
    r.confidence         = CASE row.confidence
                             WHEN "" THEN NULL
                             ELSE toFloat(row.confidence)
                           END,
    r.description        = row.rel_desc,
    r.resolution_context = row.resolution_context,
    r.justification      = row.justification,
    r.remark             = row.remark,
    r.evidence_excerpt   = row.evidence_excerpt,
    r.chunk_id           = row.chunk_id,
    // keep the pretty JSON text for inspection
    r.qualifiers_json    = row.qualifiers
  WITH r, qmap
  FOREACH (k IN keys(qmap) |
    SET r['qual_' + k] = toString(qmap[k])
  )
} IN TRANSACTIONS OF 1000 ROWS;








//#?######################### Start ##########################
//#region:#?   See classes as colors



// 1) create a sanitized label string in property `_viz_label`
MATCH (e:Entity)
WHERE e.class_group IS NOT NULL AND e.class_group <> ''
WITH e,
     replace(
       replace(
         replace(
           replace(
             replace(
               replace(
                 replace(
                   replace(
                     replace(
                       replace(e.class_group, ' ', '_'),
                     '-', '_'),
                   '/', '_'),
                 '.', '_'),
               ',', '_'),
             '"', '_'),
           "'", '_'),
         '(', '_'),
       ')', '_'),
     ':', '_') AS lab0
// ensure label begins with a letter - prefix 'C_' otherwise
WITH e,
     CASE WHEN lab0 =~ '^[A-Za-z].*' THEN lab0 ELSE 'C_' + lab0 END AS vizlabel
SET e._viz_label = vizlabel
RETURN count(e) AS nodes_updated;



// 2) add labels from _viz_label (one label per node)
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
CALL apoc.create.addLabels(e, [e._viz_label]) YIELD node
RETURN count(node) AS labels_added;


// 3) remove Entity label from nodes that have a viz label (visualization-only)
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
REMOVE e:Entity
RETURN count(e) AS entity_labels_removed;





//#endregion#? See classes as colors
//#?#########################  End  ##########################






//MATCH (n)
//RETURN n





MATCH (n)-[r]->(m)
// WHERE n.seen_in_chunks = ["Ch_000018"]
RETURN n, r, m






#endregion#? Cypher Queries
#*#########################  End  ##########################




