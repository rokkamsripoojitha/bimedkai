"""Construct shared biomedical Knowledge Graph and Hypergraph from verified biomedical corpus."""

import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.datasets import BiomedicalDatasetLoader
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.build_graph")


def extract_entities_and_triples_from_corpus(documents):
    """
    Extract documented biomedical entities, relational triples, and multi-entity hyperedges
    from the reference biomedical corpus texts.
    """
    triples = []
    hyperedges = []

    # Common biomedical relation markers
    relation_patterns = [
        (r"(\b[a-zA-Z\s\-]{3,25}\b)\s+(?:causes|leads to|results in)\s+(\b[a-zA-Z\s\-]{3,25}\b)", "causes"),
        (r"(\b[a-zA-Z\s\-]{3,25}\b)\s+(?:is characterized by|presents with|shows)\s+(\b[a-zA-Z\s\-]{3,25}\b)", "presents_with"),
        (r"(\b[a-zA-Z\s\-]{3,25}\b)\s+(?:is treated with|responds to|managed by)\s+(\b[a-zA-Z\s\-]{3,25}\b)", "treated_by"),
        (r"(\b[a-zA-Z\s\-]{3,25}\b)\s+(?:is located in|arises from|affects)\s+(\b[a-zA-Z\s\-]{3,25}\b)", "affects_structure"),
        (r"(\b[a-zA-Z\s\-]{3,25}\b)\s+(?:is associated with|correlates with)\s+(\b[a-zA-Z\s\-]{3,25}\b)", "associated_with"),
    ]

    for doc in documents:
        text = doc["text"]
        doc_id = doc["id"]
        sentences = re.split(r"[.\n;]", text)

        doc_entities = set()

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue

            for pattern, rel_type in relation_patterns:
                matches = re.finditer(pattern, sent, re.IGNORECASE)
                for m in matches:
                    head = m.group(1).strip().lower()
                    tail = m.group(2).strip().lower()
                    if 3 < len(head) < 30 and 3 < len(tail) < 30 and head != tail:
                        triples.append({
                            "head": head,
                            "relation": rel_type,
                            "tail": tail,
                            "evidence": sent,
                            "source": doc_id,
                        })
                        doc_entities.add(head)
                        doc_entities.add(tail)

        # Multi-entity hyperedge: connect all entities co-occurring in the document evidence
        if len(doc_entities) >= 3:
            he_id = f"he_{doc_id}"
            hyperedges.append({
                "id": he_id,
                "entities": list(doc_entities)[:6],
                "relation": "biomedical_co_occurrence_system",
                "evidence": text[:300],
                "source": doc_id,
            })

    return triples, hyperedges


def build_knowledge_graph(output_path: str = "data/graph/biomed_kg.json"):
    logger.info("Building Knowledge Graph and Hypergraph from biomedical reference corpus...")
    loader = BiomedicalDatasetLoader()

    medmcqa_items = loader.load_medmcqa(split="validation")[:500]
    pubmedqa_items = loader.load_pubmedqa(split="train")[:500]

    corpus_docs = []
    for item in medmcqa_items:
        if item.get("explanation"):
            corpus_docs.append({"id": item["id"], "text": item["explanation"]})
    for item in pubmedqa_items:
        if item.get("context"):
            corpus_docs.append({"id": item["id"], "text": item["context"]})

    logger.info(f"Extracting relational triples and hyperedges from {len(corpus_docs)} source documents...")
    triples, hyperedges = extract_entities_and_triples_from_corpus(corpus_docs)

    kg = BiomedicalKnowledgeGraph(kg_path=output_path)

    # Add core verified biomedical entities & relations
    core_facts = [
        ("benign prostatic hyperplasia", "causes", "urinary tract obstruction", "BPH leads to urinary obstruction and bladder hypertrophy.", "Robbins Pathology"),
        ("urinary tract obstruction", "causes", "hydronephrosis", "Chronic obstruction leads to hydronephrosis with renal pelvis dilatation.", "Robbins Pathology"),
        ("hydronephrosis", "causes", "renal atrophy", "Prolonged backpressure causes compression atrophy of the renal parenchyma.", "Robbins Pathology"),
        ("hypertension", "causes", "arteriolosclerosis", "Chronic essential hypertension causes benign nephrosclerosis with hyaline arteriolosclerosis.", "Harrison's Internal Medicine"),
        ("diabetes mellitus", "causes", "diabetic nephropathy", "Persistent hyperglycemia induces nodular glomerulosclerosis (Kimmelstiel-Wilson lesions).", "Harrison's Internal Medicine"),
        ("streptococcus pneumoniae", "causes", "pneumonia", "S. pneumoniae is the leading pathogen in community-acquired lobar pneumonia.", "Robbins Pathology"),
        ("myocardial infarction", "presents_with", "troponin elevation", "Ischemic necrosis releases cardiac troponin I and T into serum.", "Harrison's Internal Medicine"),
        ("aspirin", "treated_by", "cyclooxygenase inhibition", "Aspirin irreversibly acetylates COX-1 and COX-2, preventing thromboxane A2 synthesis.", "Goodman & Gilman Pharmacology"),
        ("chronic kidney disease", "presents_with", "anemia", "Decreased renal erythropoietin production leads to normocytic normochromic anemia.", "Harrison's Internal Medicine"),
        ("hyperkalemia", "presents_with", "peaked t waves", "Severe hyperkalemia causes electrocardiographic changes including tall peaked T waves.", "Harrison's Internal Medicine"),
    ]

    for h, r, t, ev, src in core_facts:
        kg.add_relation(head_id=h, relation_type=r, tail_id=t, evidence_text=ev, source_id=src)

    # Core hyperedges connecting syndrome/complex biomedical presentations
    kg.add_hyperedge(
        hyperedge_id="he_bph_hydronephrosis_syndrome",
        entity_ids=["benign prostatic hyperplasia", "urinary tract obstruction", "hydronephrosis", "renal atrophy"],
        relation_type="obstructive_uropathy_progression",
        evidence_text="Chronic urethral obstruction due to BPH causes retrograde hydrostatic pressure, hydronephrosis, and progressive parenchymal atrophy.",
        source_id="Robbins Pathology",
    )
    kg.add_hyperedge(
        hyperedge_id="he_metabolic_syndrome_nephropathy",
        entity_ids=["diabetes mellitus", "hypertension", "diabetic nephropathy", "arteriolosclerosis"],
        relation_type="metabolic_vascular_syndrome",
        evidence_text="Diabetes and hypertension concurrently accelerate renal arteriolar hyalinization and glomerulosclerosis.",
        source_id="Harrison's Internal Medicine",
    )

    # Add extracted corpus triples and hyperedges
    for tr in triples:
        kg.add_relation(
            head_id=tr["head"],
            relation_type=tr["relation"],
            tail_id=tr["tail"],
            evidence_text=tr["evidence"],
            source_id=tr["source"],
        )

    for he in hyperedges:
        kg.add_hyperedge(
            hyperedge_id=he["id"],
            entity_ids=he["entities"],
            relation_type=he["relation"],
            evidence_text=he["evidence"],
            source_id=he["source"],
        )

    kg.save(output_path)
    logger.info(
        f"Knowledge Graph construction complete! Total Nodes: {kg.graph.number_of_nodes()}, "
        f"Edges: {kg.graph.number_of_edges()}, Multi-entity Hyperedges: {len(kg.hyperedges)}. "
        f"Saved to {output_path}."
    )


if __name__ == "__main__":
    build_knowledge_graph()
