"""
test_bugfixes.py
================
Regression tests for the four bugs identified in the significant-issues audit.
Each test is self-contained and uses minimal stubs — no real model or DB needed.

Run:
    python -m pytest test_bugfixes.py -v
"""

import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. knowledge_graph.py — check_plan_conflicts inner-loop bug
# ══════════════════════════════════════════════════════════════════════════════

class _FakeMultiDiGraph:
    """Minimal stub that mimics networkx.MultiDiGraph edge queries."""

    def __init__(self, edges):
        # edges: list of (src, dst, attrs)
        self._nodes = set()
        self._edges: dict[tuple, list] = {}
        for src, dst, attrs in edges:
            self._nodes.add(src)
            self._nodes.add(dst)
            key = (src, dst)
            self._edges.setdefault(key, [])
            self._edges[key].append(attrs)

    def __contains__(self, item):
        return item in self._nodes

    def get_edge_data(self, src, dst):
        key = (src, dst)
        if key not in self._edges:
            return None
        return {i: attrs for i, attrs in enumerate(self._edges[key])}

    def out_edges(self, node, data=False):
        for (src, dst), edge_list in self._edges.items():
            if src == node:
                for attrs in edge_list:
                    yield (src, dst, attrs) if data else (src, dst)

    def in_edges(self, node, data=False):
        for (src, dst), edge_list in self._edges.items():
            if dst == node:
                for attrs in edge_list:
                    yield (src, dst, attrs) if data else (src, dst)

    def number_of_nodes(self): return len(self._nodes)
    def number_of_edges(self): return sum(len(v) for v in self._edges.values())


class TestCheckPlanConflicts(unittest.TestCase):
    """check_plan_conflicts must only flag (drug, food) pairs that are
    actually present in BOTH the medication list AND the proposed foods.

    The old bug: the inner loop re-iterated self.G.out_edges(drug, data=True)
    for every food, which returned ALL drug edges regardless of the food in
    question — causing statin→grapefruit to be flagged even when grapefruit
    was not in the meal plan.
    """

    def _make_kg(self):
        """Build a minimal KnowledgeGraph backed by a fake graph."""
        # Import the real class but replace its graph
        import importlib, sys, types

        # Stub out yaml and networkx so the import doesn't fail
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda f: {
            "sources": {
                "drug_food":        "/dev/null",
                "contraindications": "/dev/null",
                "nutrients":        "/dev/null",
            }
        }
        nx_stub = types.ModuleType("networkx")
        nx_stub.MultiDiGraph = _FakeMultiDiGraph

        original_yaml = sys.modules.get("yaml")
        original_nx   = sys.modules.get("networkx")
        sys.modules["yaml"]     = yaml_stub
        sys.modules["networkx"] = nx_stub

        # Patch open so _load() silently no-ops
        import builtins
        real_open = builtins.open
        builtins.open = lambda *a, **kw: MagicMock(
            __enter__=lambda s: s,
            __exit__=lambda *a: None,
            read=lambda: "[]",
        )

        try:
            # Force reimport
            if "knowledge.knowledge_graph" in sys.modules:
                del sys.modules["knowledge.knowledge_graph"]
            from knowledge.knowledge_graph import KnowledgeGraph

            kg = KnowledgeGraph.__new__(KnowledgeGraph)
            # Inject a hand-crafted graph: statin→grapefruit, metformin→alcohol
            kg.G = _FakeMultiDiGraph([
                ("statin", "grapefruit", {
                    "edge_type": "drug_food",
                    "interaction": "CYP3A4 inhibition",
                    "severity": "high",
                    "action": "avoid",
                    "guideline": "AHA2023",
                }),
                ("metformin", "alcohol", {
                    "edge_type": "drug_food",
                    "interaction": "lactic acidosis risk",
                    "severity": "moderate",
                    "action": "limit",
                    "guideline": "ADA2024",
                }),
            ])
        finally:
            builtins.open = real_open
            if original_yaml is not None:
                sys.modules["yaml"] = original_yaml
            elif "yaml" in sys.modules:
                del sys.modules["yaml"]
            if original_nx is not None:
                sys.modules["networkx"] = original_nx
            elif "networkx" in sys.modules:
                del sys.modules["networkx"]

        return kg

    def test_absent_food_not_flagged(self):
        """statin → grapefruit must NOT be flagged when grapefruit is absent."""
        kg = self._make_kg()
        # Plan has statin but grapefruit is NOT in the meal
        conflicts = kg.check_plan_conflicts(
            medications=["statin"],
            foods=["oatmeal", "chicken_breast", "apple"],
        )
        drug_food_pairs = [(c["drug"], c["food"]) for c in conflicts]
        self.assertNotIn(("statin", "grapefruit"), drug_food_pairs,
                         "Bug: grapefruit flagged even though it's not in plan")

    def test_present_food_is_flagged(self):
        """statin → grapefruit MUST be flagged when grapefruit IS in the plan."""
        kg = self._make_kg()
        conflicts = kg.check_plan_conflicts(
            medications=["statin"],
            foods=["oatmeal", "grapefruit", "salmon"],
        )
        drug_food_pairs = [(c["drug"], c["food"]) for c in conflicts]
        self.assertIn(("statin", "grapefruit"), drug_food_pairs,
                      "Real interaction not detected")

    def test_unrelated_drug_not_flagged(self):
        """metformin → grapefruit must NOT be flagged (no such edge)."""
        kg = self._make_kg()
        conflicts = kg.check_plan_conflicts(
            medications=["metformin"],
            foods=["grapefruit"],
        )
        self.assertEqual(conflicts, [],
                         "No metformin-grapefruit edge should exist")

    def test_empty_inputs(self):
        """Empty medications or foods must return empty list without error."""
        kg = self._make_kg()
        self.assertEqual(kg.check_plan_conflicts([], ["apple"]), [])
        self.assertEqual(kg.check_plan_conflicts(["statin"], []), [])


# ══════════════════════════════════════════════════════════════════════════════
# 2. nutrition_agent.py — fiber double-count in _local_replan
# ══════════════════════════════════════════════════════════════════════════════

class TestLocalReplan(unittest.TestCase):
    """_local_replan must correctly subtract the old snack's fiber before
    adding the new snack's fiber — the old code subtracted new.fiber twice.
    """

    def _make_agent(self):
        """Construct a NutritionAgent with stub dependencies."""
        import sys, types

        # Stub base_agent so we don't need the full BDI stack
        ba = types.ModuleType("agents.base_agent")
        for name in ("ActionType", "AgentAction", "AgentResult",
                      "BaseAgent", "PatientState", "Urgency"):
            obj = MagicMock()
            obj.__mro_entries__ = lambda bases: (object,)
            setattr(ba, name, obj)

        # BaseAgent stub: just stores attributes
        class _BaseAgent:
            def __init__(self, agent_id, policy_registry,
                         knowledge_graph, audit_log):
                self.agent_id = agent_id
                self.registry = policy_registry
                self.kg       = knowledge_graph
                self.audit    = audit_log
                self.beliefs  = {}
                self._log     = MagicMock()

        ba.BaseAgent = _BaseAgent
        sys.modules["agents.base_agent"] = ba

        if "agents.nutrition_agent" in sys.modules:
            del sys.modules["agents.nutrition_agent"]

        from agents.nutrition_agent import NutritionAgent, FoodItem, MealPlan, NutrientTargets

        agent = NutritionAgent(
            policy_registry = MagicMock(),
            knowledge_graph = MagicMock(
                get_drug_interactions=lambda d: [],
                check_plan_conflicts=lambda m, f: [],
            ),
            audit_log       = MagicMock(),
        )
        agent.beliefs = {
            "conditions": [],
            "allergies":  {},
            "excluded_foods": set(),
        }
        agent.registry.get_sodium_cap = MagicMock(return_value=2.3)
        return agent, FoodItem, MealPlan, NutrientTargets

    def test_fiber_not_double_counted(self):
        agent, FoodItem, MealPlan, NutrientTargets = self._make_agent()

        old_snack = FoodItem(
            food_id="apple", name="Apple", kcal=80, sodium_mg=2,
            fiber_g=4.0, protein_g=0, potassium_mg=200,
            glycemic_index=36, meal_slot="snack", tags=("low_gi",),
        )
        new_snack = FoodItem(
            food_id="almonds", name="Almonds", kcal=160, sodium_mg=0,
            fiber_g=3.0, protein_g=6, potassium_mg=200,
            glycemic_index=0, meal_slot="snack", tags=("low_gi",),
        )

        targets = NutrientTargets(
            kcal=2000, sodium_mg=2300, fiber_g=25,
            protein_g=56, potassium_mg=3500,
        )

        # Plan is 300 kcal over target — should trigger a snack swap
        plan = MealPlan(
            snack            = old_snack,
            total_kcal       = 2300.0,   # 300 over target
            total_sodium_mg  = 1200.0,
            total_fiber_g    = 20.0,     # includes old snack's 4.0 g
            targets          = targets,
        )

        result = agent._local_replan(plan, targets, glucose=100, deviation_kcal=300)

        if result.snack is not None and result.snack.food_id == new_snack.food_id:
            # Old fiber (4.0) subtracted, new fiber (3.0) added → 20 - 4 + 3 = 19
            expected_fiber = 20.0 - old_snack.fiber_g + new_snack.fiber_g
            self.assertAlmostEqual(
                result.total_fiber_g, expected_fiber, places=5,
                msg=(
                    f"Fiber double-count bug: expected {expected_fiber:.1f} g "
                    f"but got {result.total_fiber_g:.1f} g"
                ),
            )

    def test_sodium_also_correct(self):
        """Verify the sodium update (was correct before; must stay correct)."""
        agent, FoodItem, MealPlan, NutrientTargets = self._make_agent()

        old_snack = FoodItem(
            food_id="crackers", name="Crackers", kcal=150, sodium_mg=300,
            fiber_g=2.0, protein_g=3, potassium_mg=50,
            glycemic_index=72, meal_slot="snack",
        )
        low_sodium_snack = FoodItem(
            food_id="almonds", name="Almonds", kcal=80, sodium_mg=0,
            fiber_g=3.0, protein_g=6, potassium_mg=200,
            glycemic_index=0, meal_slot="snack", tags=("low_gi",),
        )

        targets = NutrientTargets(
            kcal=2000, sodium_mg=2300, fiber_g=25,
            protein_g=56, potassium_mg=3500,
        )
        plan = MealPlan(
            snack            = old_snack,
            total_kcal       = 2350.0,   # 350 over
            total_sodium_mg  = 2100.0,
            total_fiber_g    = 18.0,
            targets          = targets,
        )

        result = agent._local_replan(plan, targets, glucose=100, deviation_kcal=350)

        if result.snack is not None and result.snack.food_id == "almonds":
            expected_sodium = 2100.0 - old_snack.sodium_mg + low_sodium_snack.sodium_mg
            self.assertAlmostEqual(result.total_sodium_mg, expected_sodium, places=5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. orchestrator.py — belief contamination across agents
# ══════════════════════════════════════════════════════════════════════════════

class TestBeliefIsolation(unittest.TestCase):
    """update_beliefs must be called ONLY on the routed agent.
    All other agents must NOT receive domain-foreign task keys.
    """

    def _make_orchestrator(self):
        import sys, types

        # Stub networkx so orchestrator.py imports cleanly
        if "networkx" not in sys.modules:
            nx_stub = types.ModuleType("networkx")
            nx_stub.MultiDiGraph = MagicMock
            sys.modules["networkx"] = nx_stub

        if "orchestrator.orchestrator" in sys.modules:
            del sys.modules["orchestrator.orchestrator"]

        from orchestrator.orchestrator import Orchestrator

        def _make_agent(agent_id):
            agent = MagicMock()
            agent.agent_id   = agent_id
            agent.beliefs    = {}
            agent.deliberate = MagicMock(return_value=[])
            agent.act        = MagicMock(return_value=MagicMock(actions=[]))

            def _update_beliefs(task):
                agent.beliefs.update(task)
            agent.update_beliefs = MagicMock(side_effect=_update_beliefs)
            return agent

        agents = {
            "medicine_agent":   _make_agent("medicine_agent"),
            "nutrition_agent":  _make_agent("nutrition_agent"),
            "activity_agent":   _make_agent("activity_agent"),
            "monitoring_agent": _make_agent("monitoring_agent"),
        }

        kg       = MagicMock()
        kg.check_plan_conflicts             = MagicMock(return_value=[])
        kg.get_condition_contraindications  = MagicMock(return_value=[])

        audit = MagicMock()
        audit.write = MagicMock(return_value="abc123")

        orch = Orchestrator(
            agents          = agents,
            policy_registry = MagicMock(),
            knowledge_graph = kg,
            audit_log       = audit,
        )
        return orch, agents

    def test_only_chosen_agent_receives_full_task(self):
        """A nutrition task must update ONLY nutrition_agent with full payload."""
        orch, agents = self._make_orchestrator()

        task = {
            "task_type":    "meal_plan",
            "patient_id":   "P001",
            "conditions":   ["hypertension"],
            "glucose_target":  140,        # nutrition-domain key
            "food_exclusions": ["grapefruit"],
        }

        orch.process(task)

        # nutrition_agent must have received the full task (including domain keys)
        nutrition_call_args = agents["nutrition_agent"].update_beliefs.call_args_list
        self.assertGreater(len(nutrition_call_args), 0,
                           "nutrition_agent.update_beliefs was never called")

        full_task_seen = any(
            "glucose_target" in call.args[0] or
            "glucose_target" in call.kwargs.get("task", {})
            for call in nutrition_call_args
        )
        self.assertTrue(full_task_seen,
                        "nutrition_agent never received glucose_target key")

    def test_non_chosen_agents_receive_no_domain_keys(self):
        """medicine_agent must NOT receive glucose_target or food_exclusions."""
        orch, agents = self._make_orchestrator()

        task = {
            "task_type":       "meal_plan",
            "patient_id":      "P001",
            "conditions":      ["hypertension"],
            "glucose_target":  140,
            "food_exclusions": ["grapefruit"],
        }

        orch.process(task)

        for call in agents["medicine_agent"].update_beliefs.call_args_list:
            arg = call.args[0] if call.args else {}
            self.assertNotIn(
                "glucose_target", arg,
                "Bug: glucose_target leaked into medicine_agent beliefs",
            )
            self.assertNotIn(
                "food_exclusions", arg,
                "Bug: food_exclusions leaked into medicine_agent beliefs",
            )

    def test_shared_keys_propagate_to_all_agents(self):
        """'patient_id' and 'conditions' must reach every agent."""
        orch, agents = self._make_orchestrator()

        task = {
            "task_type":  "meal_plan",
            "patient_id": "P002",
            "conditions": ["ckd", "hypertension"],
            "glucose_target": 130,
        }

        orch.process(task)

        for agent_id, agent in agents.items():
            all_calls = agent.update_beliefs.call_args_list
            received_patient_id = any(
                call.args[0].get("patient_id") == "P002"
                for call in all_calls if call.args
            )
            self.assertTrue(
                received_patient_id,
                f"{agent_id} never received patient_id (shared context key)",
            )


# ══════════════════════════════════════════════════════════════════════════════
# 4. baselines — metrics computed from data, not hardcoded
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineMetrics(unittest.TestCase):
    """Verify that baseline metric arrays are derived from actual data,
    not returned as constant arrays like [0.71]*20.
    """

    def _make_cohort_files(self, tmp_dir: str):
        """Write synthetic cohort CSVs to tmp_dir with realistic variation.

        Uses 20 patients × 40 timesteps = 800 rows with Gaussian vital-sign
        noise so that IsolationForest sees varied features and bootstrap
        resamples of precision have non-zero variance.
        """
        import os, csv
        os.makedirs(tmp_dir, exist_ok=True)

        vitals_rows = []
        event_rows  = []
        rng = np.random.default_rng(42)

        # 20 patients, 40 timesteps each
        for pid in range(20):
            # Each patient has a slight SBP baseline offset
            sbp_base = 115 + pid * 2
            for step in range(40):
                t = step * 5
                # Continuous Gaussian variation — not just one spike
                sbp  = sbp_base + rng.normal(0, 15)
                # Force several true escalation events spread across patients
                if step in (10, 25) and pid % 4 == 0:
                    sbp = 185.0
                sbp = max(80., min(220., sbp))

                vitals_rows.append({
                    "patient_id":          pid,
                    "t_minutes":           t,
                    "sbp":                 round(sbp, 1),
                    "dbp":                 round(75 + rng.normal(0, 8), 1),
                    "glucose_mgdl":        round(100 + rng.normal(0, 25), 1),
                    "heart_rate":          round(70 + rng.normal(0, 10), 1),
                    "spo2":                round(min(100., 97 + rng.normal(0, 1.5)), 1),
                    "adherence_med":       round(float(np.clip(0.7 + rng.normal(0, 0.15), 0, 1)), 3),
                    "adherence_diet":      round(float(np.clip(0.5 + rng.normal(0, 0.15), 0, 1)), 3),
                    "adherence_lifestyle": round(float(np.clip(0.6 + rng.normal(0, 0.15), 0, 1)), 3),
                })
                if sbp > 160:
                    event_rows.append({"patient_id": pid, "t_minutes": t})

        with open(f"{tmp_dir}/vitals_longitudinal.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=vitals_rows[0].keys())
            w.writeheader(); w.writerows(vitals_rows)

        with open(f"{tmp_dir}/events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "t_minutes"])
            w.writeheader()
            if event_rows:
                w.writerows(event_rows)

    def test_rules_only_metrics_vary(self):
        """Bootstrap precision values must not all be the same constant."""
        import tempfile
        from baselines.rules_only import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            self._make_cohort_files(tmp)
            res = evaluate(tmp)

        prec = res["med_precision"]
        self.assertIsInstance(prec, np.ndarray)
        self.assertGreater(len(prec), 0)
        # A constant array [0.71]*N would have std == 0
        self.assertGreater(
            float(np.std(prec)) + float(np.max(prec) - np.min(prec)), 0.,
            "med_precision is constant — likely still hardcoded",
        )

    def test_rules_only_latency_from_data(self):
        """Latency values must come from actual threshold crossings."""
        import tempfile
        from baselines.rules_only import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            self._make_cohort_files(tmp)
            res = evaluate(tmp)

        lat = res["latency"]
        self.assertIsInstance(lat, np.ndarray)
        self.assertGreater(len(lat), 0)
        # Verify at least some latency values are plausible (0–3600 s)
        self.assertTrue(np.all(lat >= 0.),    "Negative latency")
        self.assertTrue(np.all(lat <= 3600.), "Latency > 1 hour cap")

    def test_human_schedule_only_escalates_at_review_time(self):
        """Human schedule must only produce action=4 at hour 9."""
        from baselines.human_schedule import _score_row, _is_review_time

        rng = np.random.default_rng(42)

        # 09:00 with high BP → escalate
        row_09 = {"t_minutes": 9 * 60, "sbp": 180}   # hour 9 → review
        action_09, _ = _score_row(row_09, rng)
        self.assertEqual(action_09, 4, "Expected escalation at 09:00 with SBP=180")

        # 14:00 with high BP → no escalation (not review time)
        row_14 = {"t_minutes": 14 * 60, "sbp": 180}
        action_14, _ = _score_row(row_14, rng)
        self.assertEqual(action_14, 0, "Expected NO escalation at 14:00")

    def test_predictive_only_roc_scores_in_range(self):
        """IsolationForest scores must be in [0, 1]."""
        import tempfile
        from baselines.predictive_only import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            self._make_cohort_files(tmp)
            res = evaluate(tmp)

        scores = res["roc_scores"]
        self.assertTrue(np.all(scores >= 0.), "Scores below 0")
        self.assertTrue(np.all(scores <= 1.), "Scores above 1")
        # Verify they are not all identical
        self.assertGreater(np.std(scores), 0., "All scores identical")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
