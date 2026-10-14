"""Biomedical Knowledge Graph and Hypergraph representation, traversal, and expansion."""

import os
import re
import json
import time
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx


class BiomedicalKnowledgeGraph:
    """
    Biomedical Knowledge Graph supporting entity extraction, depth-controlled traversal,
    and multi-entity hyperedge expansion for HyperGraphRAG.
    """

    def __init__(self, kg_path: Optional[str] = "data/graph/biomed_kg.json"):
        self.kg_path = kg_path
        self.graph = nx.MultiDiGraph()
        # Hypergraph structures:
        # hyperedges: Dict[hyperedge_id, {entities: Set[str], relation: str, text: str, source_id: str}]
        self.hyperedges: Dict[str, Dict[str, Any]] = {}
        # entity_to_hyperedges: Dict[entity_id, Set[hyperedge_id]]
        self.entity_to_hyperedges: Dict[str, Set[str]] = {}

        # Traversal statistics
        self.traversal_operations_count: int = 0
        self.visited_nodes_count: int = 0
        self.visited_edges_count: int = 0

        if kg_path and os.path.exists(kg_path):
            self.load(kg_path)

    def add_entity(self, entity_id: str, name: str, entity_type: str, description: str = "") -> None:
        """Add a biomedical entity node with category and attributes."""
        clean_id = entity_id.strip().lower()
        self.graph.add_node(
            clean_id,
            name=name,
            entity_type=entity_type,
            description=description,
        )

    def add_relation(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str,
        evidence_text: str = "",
        source_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Add a directed relational edge between two entities."""
        h = head_id.strip().lower()
        t = tail_id.strip().lower()
        if not self.graph.has_node(h):
            self.add_entity(h, h, "Unknown")
        if not self.graph.has_node(t):
            self.add_entity(t, t, "Unknown")

        self.graph.add_edge(
            h,
            t,
            relation=relation_type,
            evidence=evidence_text,
            source=source_id,
            confidence=confidence,
        )

    def add_hyperedge(
        self,
        hyperedge_id: str,
        entity_ids: List[str],
        relation_type: str,
        evidence_text: str,
        source_id: str = "",
    ) -> None:
        """
        Add a multi-entity hyperedge e = {v_1, v_2, ..., v_k} connecting multiple entities.
        Essential for authentic HyperGraphRAG representation.
        """
        clean_entities = set(e.strip().lower() for e in entity_ids if e.strip())
        if len(clean_entities) < 2:
            return

        self.hyperedges[hyperedge_id] = {
            "id": hyperedge_id,
            "entities": list(clean_entities),
            "relation": relation_type,
            "evidence": evidence_text,
            "source": source_id,
        }

        for ent in clean_entities:
            if ent not in self.entity_to_hyperedges:
                self.entity_to_hyperedges[ent] = set()
            self.entity_to_hyperedges[ent].add(hyperedge_id)
            if not self.graph.has_node(ent):
                self.add_entity(ent, ent, "BiomedicalEntity")

    def extract_seed_entities(self, text: str, max_seeds: int = 10) -> List[str]:
        """
        Identify seed entities from a question or context using keyword matching
        against the knowledge graph entity dictionary.
        """
        text_lower = text.lower()
        matched = []

        # Check existing nodes in graph
        for node in self.graph.nodes():
            if len(node) > 2 and re.search(rf"\b{re.escape(node)}\b", text_lower):
                matched.append(node)
                if len(matched) >= max_seeds:
                    break

        # Fallback regex extraction of key capitalized biomedical or multi-word terms if few matched
        if len(matched) < 3:
            candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for cand in candidates:
                cand_lower = cand.lower()
                if cand_lower not in matched and len(cand_lower) > 3:
                    matched.append(cand_lower)
                if len(matched) >= max_seeds:
                    break

        return matched

    def traverse_subgraph(
        self, seed_entities: List[str], depth: int = 1, max_neighbors: int = 5
    ) -> Dict[str, Any]:
        """
        Perform depth-governed traversal up to depth d (1, 2, or 3).
        Returns visited entities, relations, and traceable evidence paths.
        """
        self.traversal_operations_count += 1
        t0 = time.perf_counter()

        visited_nodes: Set[str] = set()
        visited_edges: List[Dict[str, Any]] = []
        current_frontier = set(s.strip().lower() for s in seed_entities if self.graph.has_node(s.strip().lower()))

        visited_nodes.update(current_frontier)

        for current_depth in range(1, depth + 1):
            next_frontier: Set[str] = set()
            for node in current_frontier:
                # Traverse outgoing and incoming edges
                out_edges = list(self.graph.out_edges(node, data=True))[:max_neighbors]
                in_edges = list(self.graph.in_edges(node, data=True))[:max_neighbors]

                for u, v, data in out_edges:
                    visited_edges.append({
                        "head": u,
                        "relation": data.get("relation", "relates_to"),
                        "tail": v,
                        "evidence": data.get("evidence", ""),
                        "source": data.get("source", ""),
                        "hop": current_depth,
                    })
                    if v not in visited_nodes:
                        next_frontier.add(v)
                        visited_nodes.add(v)

                for u, v, data in in_edges:
                    visited_edges.append({
                        "head": u,
                        "relation": data.get("relation", "relates_to"),
                        "tail": v,
                        "evidence": data.get("evidence", ""),
                        "source": data.get("source", ""),
                        "hop": current_depth,
                    })
                    if u not in visited_nodes:
                        next_frontier.add(u)
                        visited_nodes.add(u)

            current_frontier = next_frontier
            if not current_frontier:
                break

        elapsed = time.perf_counter() - t0
        self.visited_nodes_count += len(visited_nodes)
        self.visited_edges_count += len(visited_edges)

        return {
            "seed_entities": seed_entities,
            "depth": depth,
            "visited_nodes": list(visited_nodes),
            "visited_edges": visited_edges,
            "traversal_time": elapsed,
        }

    def expand_hypergraph(
        self, seed_entities: List[str], depth: int = 1, max_hyperedges: int = 5
    ) -> Dict[str, Any]:
        """
        Perform genuine hypergraph neighborhood expansion.
        At depth 1: retrieve incident hyperedges for seed entities.
        At depth 2: expand to hyperedges sharing entities with depth 1 hyperedges.
        At depth 3: expand to 3rd-order hyperedge neighborhood.
        """
        self.traversal_operations_count += 1
        t0 = time.perf_counter()

        active_entities = set(s.strip().lower() for s in seed_entities)
        collected_hyperedge_ids: Set[str] = set()
        expansion_trace: List[Dict[str, Any]] = []

        for current_depth in range(1, depth + 1):
            new_hyperedges: Set[str] = set()
            for ent in active_entities:
                incident = self.entity_to_hyperedges.get(ent, set())
                for he_id in incident:
                    if he_id not in collected_hyperedge_ids:
                        new_hyperedges.add(he_id)
                        collected_hyperedge_ids.add(he_id)
                        if len(new_hyperedges) >= max_hyperedges:
                            break
                if len(new_hyperedges) >= max_hyperedges:
                    break

            new_entities: Set[str] = set()
            for he_id in new_hyperedges:
                he_data = self.hyperedges[he_id]
                expansion_trace.append({
                    "hyperedge_id": he_id,
                    "depth": current_depth,
                    "entities": he_data["entities"],
                    "relation": he_data["relation"],
                    "evidence": he_data["evidence"],
                    "source": he_data["source"],
                })
                for ent in he_data["entities"]:
                    if ent not in active_entities:
                        new_entities.add(ent)

            active_entities.update(new_entities)
            if not new_entities:
                break

        elapsed = time.perf_counter() - t0
        return {
            "seed_entities": seed_entities,
            "depth": depth,
            "hyperedges": expansion_trace,
            "total_entities_covered": list(active_entities),
            "expansion_time": elapsed,
        }

    def save(self, kg_path: Optional[str] = None) -> None:
        """Save Knowledge Graph and Hypergraph to JSON."""
        p = kg_path or self.kg_path
        os.makedirs(os.path.dirname(p), exist_ok=True)

        data = {
            "nodes": [
                {"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"head": u, "tail": v, **attrs}
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "hyperedges": list(self.hyperedges.values()),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, kg_path: Optional[str] = None) -> None:
        """Load Knowledge Graph and Hypergraph from JSON."""
        p = kg_path or self.kg_path
        if not os.path.exists(p):
            raise FileNotFoundError(f"Knowledge graph file not found at {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self.hyperedges.clear()
        self.entity_to_hyperedges.clear()

        for node in data.get("nodes", []):
            nid = node.pop("id")
            self.graph.add_node(nid, **node)

        for edge in data.get("edges", []):
            h = edge.pop("head")
            t = edge.pop("tail")
            self.graph.add_edge(h, t, **edge)

        for he in data.get("hyperedges", []):
            self.add_hyperedge(
                hyperedge_id=he["id"],
                entity_ids=he["entities"],
                relation_type=he["relation"],
                evidence_text=he["evidence"],
                source_id=he.get("source", ""),
            )
