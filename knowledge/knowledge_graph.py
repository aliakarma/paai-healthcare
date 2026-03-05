"""
knowledge_graph.py
==================
Loads and queries the clinical knowledge graph.
Supports Drug-Food, Condition-Contraindication, and Nutrient-Deficiency
triples sourced from ADA 2024, AHA 2023, DrugBank v5, and WHO/USDA.

Key corrections vs. previous version
--------------------------------------
* ``check_plan_conflicts()`` contained a logic bug: the inner loop
  re-iterated ``self.G.out_edges(drug, data=True)`` instead of directly
  querying the specific ``drug → food`` edge.  This caused every drug-food
  interaction present in the graph to be flagged regardless of whether
  that food was actually in ``proposed_foods``, producing false-positive
  conflicts and potentially blocking safe meal plans.

  The corrected implementation uses ``self.G.get_edge_data(drug, food)``
  (O(1) for MultiDiGraph) to test only the exact (drug, food) pair, then
  iterates over the parallel edges at that pair to collect all matching
  interactions.  This is both correct and faster.

* ``get_drug_interactions()`` is unchanged in semantics but now explicitly
  filters on ``edge_type == "drug_food"`` to avoid accidentally returning
  contraindication or nutrient-deficiency edges that share the same source
  node.

* ``get_condition_contraindications()`` now returns the ``action`` field
  (previously omitted in some code paths) and filters strictly on
  ``edge_type == "contraindication"``.

* Module-level docstring updated to document the three edge types and their
  required attributes.

Edge types in the graph
------------------------
drug_food        : drug → food, attrs: interaction, severity, action, guideline
contraindication : drug/condition → condition/food, attrs: severity, action
nutrient_deficiency: condition/drug → nutrient, attrs: deficiency_risk, action
"""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory clinical knowledge graph backed by NetworkX MultiDiGraph.

    Nodes
    -----
    Drugs, foods, clinical conditions, nutrients — identified by canonical
    string slugs (e.g. ``"metformin"``, ``"grapefruit"``, ``"ckd"``).

    Edges
    -----
    Three typed edge sets are loaded from JSON files:

    * **drug_food** — pairwise drug-food interactions with severity grades
      (``"low"`` / ``"moderate"`` / ``"high"`` / ``"absolute"``) and a
      recommended clinical action.
    * **contraindication** — drug or condition contraindicated by a condition
      or food.
    * **nutrient_deficiency** — condition or drug associated with a nutrient
      deficiency risk.

    Parameters
    ----------
    config : optional pre-loaded config dict.  When ``None``, the graph
             reads ``configs/knowledge_graph.yaml`` automatically.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            import yaml
            with open("configs/knowledge_graph.yaml") as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load()

    # ── Graph construction ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all three JSON triple files into the in-memory graph."""
        self._load_drug_food()
        self._load_contraindications()
        self._load_nutrients()
        logger.info(
            "KnowledgeGraph loaded: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    def _load_drug_food(self) -> None:
        path = self.cfg["sources"]["drug_food"]
        with open(path) as f:
            triples = json.load(f)
        for t in triples:
            self.G.add_edge(
                t["drug"],
                t["food"],
                interaction = t["interaction"],
                severity    = t["severity"],
                action      = t["action"],
                guideline   = t.get("guideline", ""),
                edge_type   = "drug_food",
            )

    def _load_contraindications(self) -> None:
        path = self.cfg["sources"]["contraindications"]
        with open(path) as f:
            triples = json.load(f)
        for t in triples:
            src = t.get("drug") or t.get("condition", "unknown")
            dst = t.get("condition") or t.get("food", "unknown")
            self.G.add_edge(
                src, dst,
                interaction = t.get("contraindication", ""),
                severity    = t["severity"],
                action      = t.get("action", ""),
                guideline   = t.get("guideline", ""),
                edge_type   = "contraindication",
            )

    def _load_nutrients(self) -> None:
        path = self.cfg["sources"]["nutrients"]
        with open(path) as f:
            triples = json.load(f)
        for t in triples:
            src = t.get("condition") or t.get("drug", "unknown")
            self.G.add_edge(
                src,
                t["nutrient"],
                interaction = t.get("deficiency_risk", ""),
                severity    = "moderate",
                action      = f"supplement_{t['nutrient']}",
                guideline   = "",
                edge_type   = "nutrient_deficiency",
            )

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_drug_interactions(self, drug: str) -> list[dict[str, Any]]:
        """Return all known drug-food interactions for *drug*.

        Only ``edge_type == "drug_food"`` edges are returned; contraindication
        and nutrient-deficiency edges that share the source node are excluded.

        Parameters
        ----------
        drug : canonical drug name (e.g. ``"metformin"``).

        Returns
        -------
        list of dicts with keys: ``drug``, ``food``, ``interaction``,
        ``severity``, ``action``.  Empty list if drug has no interactions.
        """
        if drug not in self.G:
            return []

        results: list[dict[str, Any]] = []
        for _, target, data in self.G.out_edges(drug, data=True):
            if data.get("edge_type") == "drug_food":
                results.append({
                    "drug":        drug,
                    "food":        target,
                    "interaction": data["interaction"],
                    "severity":    data["severity"],
                    "action":      data["action"],
                    "guideline":   data.get("guideline", ""),
                })
        return results

    def get_condition_contraindications(
        self, condition: str
    ) -> list[dict[str, Any]]:
        """Return drugs contraindicated for *condition*.

        Traverses incoming ``contraindication`` edges to the condition node.

        Parameters
        ----------
        condition : condition slug, e.g. ``"ckd"``.

        Returns
        -------
        list of dicts with keys: ``drug``, ``condition``, ``severity``,
        ``action``.  Empty list if no contraindications are recorded.
        """
        if condition not in self.G:
            return []

        results: list[dict[str, Any]] = []
        for src, _, data in self.G.in_edges(condition, data=True):
            if data.get("edge_type") == "contraindication":
                results.append({
                    "drug":      src,
                    "condition": condition,
                    "severity":  data["severity"],
                    "action":    data.get("action", ""),
                    "guideline": data.get("guideline", ""),
                })
        return results

    def check_plan_conflicts(
        self,
        medications: list[str],
        foods: list[str],
    ) -> list[dict[str, Any]]:
        """Check a proposed (medication list, food list) plan for conflicts.

        **Bug fix**: the previous implementation re-iterated all out-edges of
        each drug node inside the food loop, which flagged every known drug-food
        interaction for that drug regardless of whether the food in question
        was actually in ``proposed_foods``.  For example, if a patient was
        prescribed ``"statin"`` and the plan included ``"oatmeal"``, the old
        code would still report the statin-grapefruit interaction even if
        grapefruit was absent from the plan.

        The corrected implementation:

        1. Checks the exact ``(drug, food)`` pair using
           ``self.G.get_edge_data(drug, food)``, which is O(1) in a
           MultiDiGraph and returns ``None`` when no edge exists.
        2. Iterates over the parallel edges at that pair (a dict keyed by
           integer edge-key) to collect all matching ``drug_food`` edges.
        3. Only flags interactions where the food is genuinely in
           ``proposed_foods``.

        Parameters
        ----------
        medications   : list of canonical drug names in the current regimen.
        foods         : list of food IDs in the proposed meal plan.

        Returns
        -------
        list of conflict dicts with keys: ``drug``, ``food``, ``severity``,
        ``action``.  Empty list means the plan is free of known conflicts.
        """
        if not medications or not foods:
            return []

        conflicts: list[dict[str, Any]] = []

        for drug in medications:
            for food in foods:
                # O(1) direct edge lookup — returns None when no edge exists
                edge_data = self.G.get_edge_data(drug, food)
                if edge_data is None:
                    continue  # no edge between this (drug, food) pair at all

                # edge_data is a dict of {edge_key: attr_dict} for MultiDiGraph
                for _key, attrs in edge_data.items():
                    if attrs.get("edge_type") != "drug_food":
                        continue   # skip contraindication / nutrient edges
                    conflicts.append({
                        "drug":        drug,
                        "food":        food,
                        "interaction": attrs.get("interaction", ""),
                        "severity":    attrs.get("severity", "low"),
                        "action":      attrs.get("action", "monitor"),
                        "guideline":   attrs.get("guideline", ""),
                    })

        return conflicts

    # ── Utility ───────────────────────────────────────────────────────────────

    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        """Return total number of edges (all types combined)."""
        return self.G.number_of_edges()

    def has_interaction(self, drug: str, food: str) -> bool:
        """Return True if a drug-food interaction edge exists for (drug, food).

        Parameters
        ----------
        drug : canonical drug name
        food : food identifier
        """
        edge_data = self.G.get_edge_data(drug, food)
        if edge_data is None:
            return False
        return any(
            attrs.get("edge_type") == "drug_food"
            for attrs in edge_data.values()
        )
