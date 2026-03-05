"""
knowledge_graph.py
==================
Loads and queries the clinical knowledge graph.
Supports Drug-Food, Condition-Contraindication, and Nutrient-Deficiency triples.
"""
import json
import networkx as nx
from pathlib import Path


class KnowledgeGraph:
    """
    In-memory knowledge graph backed by NetworkX.
    Nodes: drugs, foods, conditions, nutrients.
    Edges: interactions with severity and guideline attributes.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            import yaml
            with open("configs/knowledge_graph.yaml") as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.G = nx.MultiDiGraph()
        self._load()

    def _load(self):
        base = Path(self.cfg["sources"]["drug_food"]).parent

        with open(self.cfg["sources"]["drug_food"]) as f:
            for triple in json.load(f):
                self.G.add_edge(
                    triple["drug"], triple["food"],
                    interaction=triple["interaction"],
                    severity=triple["severity"],
                    action=triple["action"],
                    guideline=triple.get("guideline", ""),
                    edge_type="drug_food")

        with open(self.cfg["sources"]["contraindications"]) as f:
            for triple in json.load(f):
                self.G.add_edge(
                    triple.get("drug", triple.get("condition")),
                    triple.get("condition", triple.get("food")),
                    interaction=triple.get("contraindication", ""),
                    severity=triple["severity"],
                    action=triple.get("action", ""),
                    guideline=triple.get("guideline", ""),
                    edge_type="contraindication")

        with open(self.cfg["sources"]["nutrients"]) as f:
            for triple in json.load(f):
                src = triple.get("condition") or triple.get("drug")
                self.G.add_edge(
                    src, triple["nutrient"],
                    interaction=triple.get("deficiency_risk", ""),
                    severity="moderate",
                    action=f"supplement_{triple['nutrient']}",
                    guideline="",
                    edge_type="nutrient_deficiency")

    def get_drug_interactions(self, drug: str) -> list[dict]:
        """Return all known interactions for a drug."""
        results = []
        for _, target, data in self.G.out_edges(drug, data=True):
            if data.get("edge_type") == "drug_food":
                results.append({
                    "drug": drug, "food": target,
                    "interaction": data["interaction"],
                    "severity": data["severity"],
                    "action": data["action"],
                })
        return results

    def get_condition_contraindications(self, condition: str) -> list[dict]:
        """Return contraindicated drugs for a condition."""
        results = []
        for src, _, data in self.G.in_edges(condition, data=True):
            if data.get("edge_type") == "contraindication":
                results.append({
                    "drug": src, "condition": condition,
                    "severity": data["severity"],
                    "action": data["action"],
                })
        return results

    def check_plan_conflicts(self, medications: list[str],
                              foods: list[str]) -> list[dict]:
        """
        Check a proposed medication + food plan for conflicts.
        Returns list of conflict dicts (may be empty = safe plan).
        """
        conflicts = []
        for drug in medications:
            for food in foods:
                if self.G.has_edge(drug, food):
                    for _, _, data in self.G.out_edges(drug, data=True):
                        if _ == drug and data.get("edge_type") == "drug_food":
                            conflicts.append({
                                "drug": drug, "food": food,
                                "severity": data["severity"],
                                "action": data["action"],
                            })
        return conflicts
