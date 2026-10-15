"""Biomedical Knowledge Graph and Hypergraph representation, traversal, and expansion."""

import os
import re
import csv
import json
import time
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx


class BiomedicalKnowledgeGraph:
    """
    Biomedical Knowledge Graph supporting entity extraction, depth-controlled traversal,
    multi-entity hyperedge expansion, O(1)/O(M) dictionary-based entity lookup,
    and full PrimeKG dataset integration.
    """

    # Generic English stop words to filter out from single-token (1-gram) seed matches
    GENERIC_STOPWORDS = {
        "the", "and", "is", "of", "in", "to", "for", "with", "or", "an", "a", "not",
        "by", "from", "at", "as", "on", "are", "was", "were", "be", "been", "has",
        "have", "had", "do", "does", "did", "can", "could", "should", "would",
        "may", "might", "must", "which", "what", "where", "when", "how", "that",
        "this", "these", "those", "patient", "clinical", "disease", "treatment",
        "structure", "change", "associated", "leads", "following", "due", "because",
    }

    def __init__(self, kg_path: Optional[str] = "data/graph/biomed_kg.json"):
        self.kg_path = kg_path
        self.graph = nx.MultiDiGraph()
        # Hypergraph structures:
        # hyperedges: Dict[hyperedge_id, {entities: List[str], relation: str, evidence: str, source: str}]
        self.hyperedges: Dict[str, Dict[str, Any]] = {}
        # entity_to_hyperedges: Dict[entity_id, Set[hyperedge_id]]
        self.entity_to_hyperedges: Dict[str, Set[str]] = {}

        # Traversal statistics
        self.traversal_operations_count: int = 0
        self.visited_nodes_count: int = 0
        self.visited_edges_count: int = 0

        # O(1) / O(M) Entity Lookup Index
        # Maps normalized name/alias -> Set[canonical_node_id]
        self._name_to_node_ids: Dict[str, Set[str]] = {}
        self._indexed_names_count: int = 0
        self._index_built: bool = False
        self._max_ngram_len: int = 6

        if kg_path and os.path.exists(kg_path):
            self.load(kg_path)

    @staticmethod
    def normalize_entity_name(text: str) -> str:
        """
        Normalize entity name or text query for consistent lookup.
        Converts to lowercase, normalizes whitespace and punctuation while
        preserving alphanumeric characters and meaningful hyphens.
        """
        if not text:
            return ""
        # Replace non-alphanumeric and non-hyphen chars with space
        cleaned = re.sub(r"[^\w\s\-]", " ", str(text).lower())
        # Collapse multiple spaces
        return re.sub(r"\s+", " ", cleaned).strip()

    def _index_name_entry(self, name: str, node_id: str) -> None:
        """Helper to index a single normalized name string to a node ID."""
        norm_name = self.normalize_entity_name(name)
        if len(norm_name) < 2:
            return

        if norm_name not in self._name_to_node_ids:
            self._name_to_node_ids[norm_name] = set()
        self._name_to_node_ids[norm_name].add(node_id)

        # If name contains hyphens, also index the space-separated variant (e.g. 'covid-19' -> 'covid 19')
        if "-" in norm_name:
            unhyphenated = norm_name.replace("-", " ")
            unhyphenated = re.sub(r"\s+", " ", unhyphenated).strip()
            if unhyphenated and unhyphenated != norm_name:
                if unhyphenated not in self._name_to_node_ids:
                    self._name_to_node_ids[unhyphenated] = set()
                self._name_to_node_ids[unhyphenated].add(node_id)

    def build_entity_lookup_index(self) -> None:
        """
        Build an efficient in-memory lookup index mapping normalized entity names,
        canonical IDs, and aliases to graph node IDs.
        Avoids O(N) regex scanning over all graph nodes on every question.
        """
        self._name_to_node_ids.clear()

        for node_id, data in self.graph.nodes(data=True):
            clean_id = str(node_id).strip().lower()
            self._index_name_entry(clean_id, clean_id)

            # Index human-readable name attribute
            node_name = data.get("name")
            if node_name:
                self._index_name_entry(node_name, clean_id)

            # Index aliases / synonyms if present
            aliases = data.get("aliases", [])
            if isinstance(aliases, (list, set, tuple)):
                for alias in aliases:
                    if alias:
                        self._index_name_entry(str(alias), clean_id)

        self._indexed_names_count = len(self._name_to_node_ids)
        self._index_built = True

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str = "",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Add a biomedical entity node with category, attributes, and optional aliases."""
        clean_id = str(entity_id).strip().lower()
        alias_list = [str(a).strip() for a in (aliases or []) if a]
        self.graph.add_node(
            clean_id,
            name=name,
            entity_type=entity_type,
            description=description,
            aliases=alias_list,
        )

        # Incrementally update lookup index if already built
        if self._index_built:
            self._index_name_entry(clean_id, clean_id)
            if name:
                self._index_name_entry(name, clean_id)
            for a in alias_list:
                self._index_name_entry(a, clean_id)
            self._indexed_names_count = len(self._name_to_node_ids)

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
        h = str(head_id).strip().lower()
        t = str(tail_id).strip().lower()
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
        clean_entities = set(str(e).strip().lower() for e in entity_ids if str(e).strip())
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
        Identify seed entities from a question or context using an efficient dictionary-based
        multi-word n-gram matcher.
        Guarantees O(M) search complexity in question length M rather than O(N) scanning
        over all graph nodes.
        Preserves word boundaries, prevents duplicate seeds, and validates that every
        returned seed entity actually exists in the graph.
        """
        if not self._index_built:
            self.build_entity_lookup_index()

        norm_text = self.normalize_entity_name(text)
        if not norm_text:
            return []

        words = norm_text.split()
        num_words = len(words)
        if num_words == 0:
            return []

        matched_seeds: List[str] = []
        seen_seeds: Set[str] = set()

        # Generate n-grams from longest (max_ngram_len) down to 1 (longest match first)
        max_n = min(num_words, self._max_ngram_len)
        for n in range(max_n, 0, -1):
            for i in range(num_words - n + 1):
                ngram = " ".join(words[i : i + n])

                # Filter out generic 1-gram stop words
                if n == 1 and ngram in self.GENERIC_STOPWORDS:
                    continue

                if ngram in self._name_to_node_ids:
                    candidate_ids = self._name_to_node_ids[ngram]
                    for cid in candidate_ids:
                        if cid not in seen_seeds and self.graph.has_node(cid):
                            seen_seeds.add(cid)
                            matched_seeds.append(cid)
                            if len(matched_seeds) >= max_seeds:
                                return matched_seeds

        # Fallback: if no seeds matched via words, search for capitalized multi-word terms
        if len(matched_seeds) < 2:
            capitalized_candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for cand in capitalized_candidates:
                norm_cand = self.normalize_entity_name(cand)
                if norm_cand and norm_cand in self._name_to_node_ids:
                    for cid in self._name_to_node_ids[norm_cand]:
                        if cid not in seen_seeds and self.graph.has_node(cid):
                            seen_seeds.add(cid)
                            matched_seeds.append(cid)
                            if len(matched_seeds) >= max_seeds:
                                break
                if len(matched_seeds) >= max_seeds:
                    break

        return matched_seeds

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
        current_frontier = set(str(s).strip().lower() for s in seed_entities if self.graph.has_node(str(s).strip().lower()))

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

        active_entities = set(str(s).strip().lower() for s in seed_entities if self.graph.has_node(str(s).strip().lower()))
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

    def load_primekg_csv(
        self,
        csv_path: str = "data/raw/primekg.csv",
        max_rows: Optional[int] = None,
        build_hyperedges: bool = True,
    ) -> Dict[str, Any]:
        """
        Load the full Precision Medicine Knowledge Graph (PrimeKG) from CSV.
        Auto-detects column headers to robustly support Harvard PrimeKG releases:
        ['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name', 'x_source',
         'y_index', 'y_id', 'y_type', 'y_name', 'y_source']
        Maps canonical IDs, relation types, and entity metadata into NetworkX graph and hypergraph.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"PrimeKG dataset file not found at '{csv_path}'.\n"
                "Please download the official PrimeKG dataset from Harvard Dataverse:\n"
                "  URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXACMR\n"
                f"or place 'kg.csv' / 'primekg.csv' inside '{os.path.dirname(csv_path)}/'.\n"
                "To run with the small verified reference graph for testing, use the --smoke-test flag."
            )

        t0 = time.perf_counter()
        total_rows_read = 0
        valid_edges_added = 0
        skipped_rows = 0

        # Hyperedge clustering buffer: disease_id -> Set[related_entity_ids]
        disease_clusters: Dict[str, Set[str]] = {}

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"PrimeKG file '{csv_path}' is empty.")

            # Header mapping: identify column indices
            header_lower = [h.strip().lower() for h in header]
            col_map = {name: idx for idx, name in enumerate(header_lower)}

            # Identify source node columns
            idx_x_id = col_map.get("x_id") or col_map.get("source_id") or col_map.get("head_id") or col_map.get("x_index")
            idx_x_name = col_map.get("x_name") or col_map.get("source_name") or col_map.get("head_name") or idx_x_id
            idx_x_type = col_map.get("x_type") or col_map.get("source_type") or col_map.get("head_type")
            idx_x_source = col_map.get("x_source") or col_map.get("source")

            # Identify target node columns
            idx_y_id = col_map.get("y_id") or col_map.get("target_id") or col_map.get("tail_id") or col_map.get("y_index")
            idx_y_name = col_map.get("y_name") or col_map.get("target_name") or col_map.get("tail_name") or idx_y_id
            idx_y_type = col_map.get("y_type") or col_map.get("target_type") or col_map.get("tail_type")
            idx_y_source = col_map.get("y_source")

            # Identify relation column
            idx_rel = col_map.get("display_relation") or col_map.get("relation") or col_map.get("edge_type") or col_map.get("predicate")

            if idx_x_id is None or idx_y_id is None or idx_rel is None:
                raise ValueError(
                    f"Incompatible PrimeKG CSV header: {header}.\n"
                    "Required columns: ('x_id'/'source_id', 'y_id'/'target_id', 'relation'/'display_relation')."
                )

            for row in reader:
                total_rows_read += 1
                if max_rows and total_rows_read > max_rows:
                    break

                if len(row) <= max(idx_x_id, idx_y_id, idx_rel):
                    skipped_rows += 1
                    continue

                x_id_raw = row[idx_x_id].strip()
                y_id_raw = row[idx_y_id].strip()
                rel_raw = row[idx_rel].strip()

                if not x_id_raw or not y_id_raw or not rel_raw:
                    skipped_rows += 1
                    continue

                x_name = row[idx_x_name].strip() if idx_x_name is not None and len(row) > idx_x_name else x_id_raw
                x_type = row[idx_x_type].strip() if idx_x_type is not None and len(row) > idx_x_type else "BiomedicalEntity"
                x_source = row[idx_x_source].strip() if idx_x_source is not None and len(row) > idx_x_source else "PrimeKG"

                y_name = row[idx_y_name].strip() if idx_y_name is not None and len(row) > idx_y_name else y_id_raw
                y_type = row[idx_y_type].strip() if idx_y_type is not None and len(row) > idx_y_type else "BiomedicalEntity"
                y_source = row[idx_y_source].strip() if idx_y_source is not None and len(row) > idx_y_source else "PrimeKG"

                # Prefer canonical stable identifier; if numeric index only, combine with name
                clean_x_id = f"primekg_{x_id_raw}".lower() if x_id_raw.isdigit() else x_id_raw.lower()
                clean_y_id = f"primekg_{y_id_raw}".lower() if y_id_raw.isdigit() else y_id_raw.lower()

                # Add entities with metadata and names
                if not self.graph.has_node(clean_x_id):
                    self.add_entity(clean_x_id, name=x_name, entity_type=x_type, description=f"Source: {x_source}")
                if not self.graph.has_node(clean_y_id):
                    self.add_entity(clean_y_id, name=y_name, entity_type=y_type, description=f"Source: {y_source}")

                # Add directed edge
                evidence = f"{x_name} -[{rel_raw}]-> {y_name}"
                self.add_relation(
                    head_id=clean_x_id,
                    relation_type=rel_raw,
                    tail_id=clean_y_id,
                    evidence_text=evidence,
                    source_id="PrimeKG",
                )
                valid_edges_added += 1

                # Hypergraph multi-entity grouping
                if build_hyperedges:
                    if x_type.lower() == "disease":
                        if clean_x_id not in disease_clusters:
                            disease_clusters[clean_x_id] = set()
                        disease_clusters[clean_x_id].add(clean_y_id)
                    elif y_type.lower() == "disease":
                        if clean_y_id not in disease_clusters:
                            disease_clusters[clean_y_id] = set()
                        disease_clusters[clean_y_id].add(clean_x_id)

        # Build hyperedges from disease clusters having >= 3 related entities
        hyperedges_count = 0
        if build_hyperedges:
            for dis_id, related_ents in disease_clusters.items():
                if len(related_ents) >= 3:
                    cluster_members = [dis_id] + list(related_ents)[:7]
                    he_id = f"he_primekg_{dis_id}"
                    dis_name = self.graph.nodes[dis_id].get("name", dis_id)
                    self.add_hyperedge(
                        hyperedge_id=he_id,
                        entity_ids=cluster_members,
                        relation_type="primekg_disease_indication_cluster",
                        evidence_text=f"PrimeKG multi-entity syndrome cluster for {dis_name}.",
                        source_id="PrimeKG",
                    )
                    hyperedges_count += 1

        # Rebuild entity lookup index over all loaded nodes
        self.build_entity_lookup_index()
        elapsed = time.perf_counter() - t0

        return {
            "source_file": csv_path,
            "total_rows_read": total_rows_read,
            "nodes_added": self.graph.number_of_nodes(),
            "edges_added": valid_edges_added,
            "hyperedges_added": len(self.hyperedges),
            "skipped_rows": skipped_rows,
            "indexed_names_count": self._indexed_names_count,
            "elapsed_seconds": elapsed,
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
        """Load Knowledge Graph and Hypergraph from JSON and build the entity lookup index."""
        p = kg_path or self.kg_path
        if not os.path.exists(p):
            raise FileNotFoundError(f"Knowledge graph file not found at {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self.hyperedges.clear()
        self.entity_to_hyperedges.clear()
        self._name_to_node_ids.clear()
        self._index_built = False

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

        # Build lookup index immediately after loading
        self.build_entity_lookup_index()
