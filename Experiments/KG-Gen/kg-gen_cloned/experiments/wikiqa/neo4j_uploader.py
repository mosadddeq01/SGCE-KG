"""
Neo4j uploader (experiments)

This experimental script shows how to upload generated knowledge graphs to Neo4j
(both local and Neo4j AuraDB). Prefer using this from notebooks or integrating
into larger experiment flows. Keep this under experiments until validated on
large graphs.

Quick usage
-----------

You can upload your generated knowledge graphs to Neo4j databases (both local and
cloud instances like Neo4j AuraDB):

Example (from a notebook):

    from kg_gen.utils.neo4j_integration import upload_to_neo4j, Neo4jUploader

    # Simple upload to Neo4j
    success = upload_to_neo4j(
        graph=graph_1,
        uri="bolt://localhost:7687",  # or neo4j+s://<instance>.databases.neo4j.io for AuraDB
        username="neo4j",
        password="your-password",
        graph_name="my_graph",
        clear_existing=True
    )

    # Or use the uploader class for more control
    uploader = Neo4jUploader(uri="bolt://localhost:7687", username="neo4j", password="your-password")
    if uploader.connect():
        uploader.upload_graph(graph_1, graph_name="family_graph")

        # Query your uploaded graph
        stats = uploader.get_graph_stats()
        print(f"Nodes: {stats['node_count']}, Relationships: {stats['relationship_count']}")

        uploader.close()

Mapping to Neo4j
----------------
- Entities → Nodes with label `Entity`
- Relations → Relationships with the predicate as the relationship type
- Graph Name → Optional label/property you can set for organization

Notes
-----
- For AuraDB, use a URI like: neo4j+s://<instance-id>.databases.neo4j.io
- Validate on large graphs before promoting into the core package
"""

from typing import Optional

try:
    # Optional import; this script is a thin wrapper used in experiments
    from kg_gen.utils.neo4j_integration import upload_to_neo4j  # type: ignore
except Exception:
    upload_to_neo4j = None  # Fallback if the core utility isn't available


def upload_graph_experiment(
    graph,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    graph_name: Optional[str] = None,
    clear_existing: bool = False,
) -> bool:
    """Minimal helper for experiments to upload a graph to Neo4j.

    This defers to kg_gen.utils.neo4j_integration.upload_to_neo4j if present.
    Returns False if the utility isn't available in the environment.
    """
    if upload_to_neo4j is None:
        return False
    return upload_to_neo4j(
        graph=graph,
        uri=uri,
        username=username,
        password=password,
        database=database,
        graph_name=graph_name,
        clear_existing=clear_existing,
    )
