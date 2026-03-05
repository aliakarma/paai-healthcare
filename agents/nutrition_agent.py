"""
nutrition_agent.py
==================
Listing 3 from the paper — Food & Nutrition Agent.

Responsible for:
    * Building daily meal plans that satisfy macro/micronutrient targets.
    * Respecting sodium caps, potassium restrictions (CKD), glycemic targets.
    * Filtering by allergy exclusions and drug-food contraindications.
    * Issuing *local swaps* instead of wholesale re-plans for stability.

**Strategy:** The agent uses a ranked-candidate selection per meal slot,
scoring each food by how well it fits the patient's current clinical state
(post-prandial glucose, remaining sodium budget, fibre targets).  When the
daily caloric total deviates from the target by > 300 kcal, a local swap of
the snack slot is performed rather than re-planning from scratch.

Architecture position::

    preprocessing → envs → agents → orchestrator
                                     ↑
                             nutrition_agent.py (here)

Imports::

    from paai_healthcare.agents.nutrition_agent import NutritionAgent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import (
    ActionType,
    AgentAction,
    AgentResult,
    BaseAgent,
    MedicationEntry,       # Issue 4: was missing — needed for state.prescriptions type
    PatientState,
    Urgency,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Food database  (replace with USDA FoodData Central loader in production)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FoodItem:
    """Immutable representation of a single food in the database.

    Attributes:
        food_id:        Slug identifier (e.g. ``"oatmeal"``).
        name:           Human-readable display name.
        kcal:           Energy per serving (kcal).
        sodium_mg:      Sodium per serving (mg).
        fiber_g:        Dietary fibre per serving (g).
        protein_g:      Protein per serving (g).
        potassium_mg:   Potassium per serving (mg).
        glycemic_index: Glycaemic index (0 = not applicable, e.g. meat).
        meal_slot:      Primary meal slot: ``"breakfast"``, ``"lunch"``,
                        ``"dinner"``, or ``"snack"``.
        tags:           Semantic tags for filtering (e.g. ``"potassium_rich"``).
    """

    food_id:        str
    name:           str
    kcal:           float
    sodium_mg:      float
    fiber_g:        float
    protein_g:      float
    potassium_mg:   float
    glycemic_index: int
    meal_slot:      str
    tags:           tuple[str, ...] = field(default_factory=tuple)


# Built-in food database (10 items — real deployment loads from USDA/custom DB)
_FOOD_DB: tuple[FoodItem, ...] = (
    FoodItem("oatmeal",       "Steel-cut oatmeal",   150,  5, 4.0,  5, 150, 55,
             "breakfast", ("low_gi", "high_fiber")),
    FoodItem("greek_yogurt",  "Plain Greek yogurt",  130, 60, 0.0, 17, 240, 11,
             "breakfast", ("high_protein", "probiotic")),
    FoodItem("grilled_salmon","Grilled salmon fillet",280, 80, 0.0, 35, 500,  0,
             "lunch",     ("high_protein", "omega3")),
    FoodItem("lentil_soup",   "Red lentil soup",     180,250, 8.0, 12, 370, 29,
             "lunch",     ("high_fiber", "plant_protein", "low_gi")),
    FoodItem("spinach_salad", "Spinach side salad",   50, 60, 3.0,  3, 400, 15,
             "lunch",     ("low_gi", "low_calorie", "potassium_rich")),
    FoodItem("chicken_breast","Grilled chicken breast",250,75, 0.0, 40, 300,  0,
             "dinner",    ("high_protein", "low_fat")),
    FoodItem("brown_rice",    "Cooked brown rice",   220,  5, 3.5,  5,  84, 68,
             "dinner",    ("whole_grain",)),
    FoodItem("sweet_potato",  "Baked sweet potato",  120, 40, 4.0,  2, 450, 63,
             "dinner",    ("vitamin_a", "potassium_rich")),
    FoodItem("apple",         "Medium apple",         80,  2, 4.0,  0, 200, 36,
             "snack",     ("low_gi", "portable")),
    FoodItem("almonds",       "Dry-roasted almonds", 160,  0, 3.0,  6, 200,  0,
             "snack",     ("healthy_fat", "low_gi")),
)


@dataclass
class NutrientTargets:
    """Per-patient daily macro/micronutrient targets.

    Computed by :meth:`NutritionAgent._compute_targets` from BMI, conditions,
    and the policy registry's sodium cap.

    Attributes:
        kcal:        Total daily energy target (kcal).
        sodium_mg:   Maximum daily sodium (mg).
        fiber_g:     Minimum daily fibre (g).
        protein_g:   Minimum daily protein (g).
        potassium_mg: Minimum daily potassium (mg) — restricted for CKD.
    """

    kcal:         float = 2000.0
    sodium_mg:    float = 2300.0
    fiber_g:      float = 25.0
    protein_g:    float = 56.0
    potassium_mg: float = 3500.0


@dataclass
class MealPlan:
    """A complete daily meal plan produced by the Nutrition Agent.

    Attributes:
        breakfast:      Selected breakfast :class:`FoodItem` (or ``None``).
        lunch:          Selected lunch :class:`FoodItem` (or ``None``).
        dinner:         Selected dinner :class:`FoodItem` (or ``None``).
        snack:          Selected snack :class:`FoodItem` (or ``None``).
        total_kcal:     Sum of energy across all slots.
        total_sodium_mg: Sum of sodium across all slots.
        total_fiber_g:  Sum of fibre across all slots.
        targets:        The :class:`NutrientTargets` used for this plan.
        notes:          Free-text notes (e.g. swap rationale).
    """

    breakfast:       FoodItem | None = None
    lunch:           FoodItem | None = None
    dinner:          FoodItem | None = None
    snack:           FoodItem | None = None
    total_kcal:      float           = 0.0
    total_sodium_mg: float           = 0.0
    total_fiber_g:   float           = 0.0
    targets:         NutrientTargets = field(default_factory=NutrientTargets)
    notes:           list[str]       = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# NutritionAgent
# ──────────────────────────────────────────────────────────────────────────────

class NutritionAgent(BaseAgent):
    """BDI Food & Nutrition Agent — daily meal planning with clinical constraints.

    **BDI mental state:**

    * Beliefs  — vitals (glucose, BP), conditions, allergies, BMI,
                 proposed drugs, remaining sodium budget.
    * Desires  — meet caloric target, stay below sodium cap, meet fibre and
                 potassium targets, maintain low post-prandial glucose.
    * Intentions — build_meal_plan, swap_item (local replan).

    **Meal-slot selection algorithm (Listing 3):**

    For each slot in ``[breakfast, lunch, dinner, snack]``:

    1. ``CandidateMeals(D, τ)``   — filter by meal slot.
    2. ``FilterByPolicy(C, P)``   — exclude allergy/condition tags.
    3. ``RankByFit(C′, x, τ)``    — score by glucose state + sodium budget.
    4. Select top-ranked candidate.

    After all slots are filled, check caloric deviation.
    If ``|total_kcal − target| > 300``: call ``LocalReplan(M, τ)`` on snack.

    Args:
        policy_registry:   Registry for sodium caps and timing windows.
        knowledge_graph:   Clinical KG (used to check drug-food exclusions).
        audit_log:         Hash-chained audit sink.
        food_db:           Override the built-in food database.  Each entry must
                           be a :class:`FoodItem`.
    """

    _MEAL_SLOTS = ("breakfast", "lunch", "dinner", "snack")

    _DEFAULT_DESIRES = [
        "meet_caloric_target",
        "sodium_below_daily_cap",
        "maximise_dietary_fibre",
        "adequate_potassium_intake",
        "minimise_post_prandial_glucose",
    ]

    def __init__(
        self,
        policy_registry: Any,
        knowledge_graph: Any,
        audit_log: Any,
        food_db: tuple[FoodItem, ...] | None = None,
    ) -> None:
        super().__init__(
            agent_id        = "nutrition_agent",
            policy_registry = policy_registry,
            knowledge_graph = knowledge_graph,
            audit_log       = audit_log,
        )
        self._food_db: tuple[FoodItem, ...] = food_db or _FOOD_DB

    # ── BDI cycle ─────────────────────────────────────────────────────────────

    def perceive(self, state: PatientState) -> None:
        """Ingest patient state and populate nutrition-domain beliefs.

        Beliefs populated:

        * ``patient_id``, ``vitals``, ``conditions``, ``allergies``
        * ``bmi``            — body mass index (from ``state.extra`` or default)
        * ``prescriptions``  — used to detect drug-food exclusions
        * ``adherence_diet`` — dietary adherence fraction
        * ``excluded_foods`` — foods excluded by KG drug-food checks

        Args:
            state: Current patient state from the orchestrator.
        """
        vitals_dict = {
            "glucose_mgdl": (state.vitals.glucose_mgdl or 100.0),
            "sbp":          (state.vitals.sbp or 120.0),
            "dbp":          (state.vitals.dbp or 80.0),
        }

        # Derive drug-food exclusions from KG
        excluded_foods: set[str] = set()
        for med in state.prescriptions:
            for interaction in self.kg.get_drug_interactions(med.drug):
                if interaction.get("severity") in ("high", "absolute"):
                    food = interaction.get("food", "")
                    if food:
                        excluded_foods.add(food)

        self.beliefs.update({
            "patient_id":    state.patient_id,
            "vitals":        vitals_dict,
            "conditions":    state.conditions,
            "allergies":     state.allergies,
            "prescriptions": state.prescriptions,
            "bmi":           state.extra.get("bmi", 27.0),
            "weight_kg":     state.extra.get("weight_kg"),
            "adherence_diet": state.adherence_diet,
            "excluded_foods": excluded_foods,
        })
        self._log.debug(
            "Perceived: patient=%s glucose=%.1f conditions=%s",
            state.patient_id,
            vitals_dict["glucose_mgdl"],
            state.conditions,
        )

    def deliberate(self) -> list[dict]:
        """Generate a ``build_meal_plan`` intention.

        If dietary adherence is critically low (< 0.35), also appends an
        ``adherence_intervention`` intention that the orchestrator may route
        to the clinician dashboard.

        Returns:
            List with at least one ``build_meal_plan`` intention.
        """
        intentions: list[dict] = [
            {"type": ActionType.MEAL_PLAN.value, "urgency": Urgency.ROUTINE.value}
        ]
        adherence = self.beliefs.get("adherence_diet", 0.55)
        if adherence < 0.35:
            intentions.append({
                "type":    "dietary_adherence_alert",
                "urgency": Urgency.HIGH.value,
                "message": (
                    f"Dietary adherence critically low ({adherence:.0%}). "
                    "Consider a patient education session."
                ),
            })
        return intentions

    def act(self, intentions: list[dict]) -> AgentResult:
        """Build the meal plan and package it as an :class:`AgentAction`.

        For each intention:

        * ``MEAL_PLAN`` → run the full slot-selection pipeline, apply local
          replan if needed, return :class:`AgentAction` with a serialised
          :class:`MealPlan` in the payload.
        * ``dietary_adherence_alert`` → return a HIGH-urgency info action.

        Args:
            intentions: List of intention dicts from :meth:`deliberate`.

        Returns:
            :class:`AgentResult` with meal plan action(s).
        """
        actions: list[AgentAction] = []

        for intent in intentions:
            intent_type = intent.get("type", "")

            if intent_type == ActionType.MEAL_PLAN.value:
                plan = self._build_plan()
                action = self._make_action(
                    action_type = ActionType.MEAL_PLAN,
                    urgency     = Urgency.ROUTINE,
                    payload     = self._serialise_plan(plan),
                    rationale   = (
                        f"Daily meal plan built — "
                        f"estimated {plan.total_kcal:.0f} kcal, "
                        f"{plan.total_sodium_mg:.0f} mg sodium."
                    ),
                )
                if self._safety_gate(action):
                    actions.append(action)

            elif intent_type == "dietary_adherence_alert":
                action = self._make_action(
                    action_type = ActionType.DIETARY_MODIFICATION,   # was DIETARY_RESTRICTION (Issue 2)
                    urgency     = Urgency.HIGH,
                    payload     = {"message": intent.get("message", "")},
                    rationale   = "Low dietary adherence flag for clinician review.",
                )
                actions.append(action)

        self._persist_actions(actions)
        return AgentResult(
            agent_id = self.agent_id,
            actions  = actions,
            metadata = {
                "adherence_diet": self.beliefs.get("adherence_diet"),
                "conditions":     self.beliefs.get("conditions"),
            },
        )

    # ── Legacy dict interface (Issue 1 fix) ──────────────────────────────────

    def execute(self, task: dict) -> dict:
        """Execute one nutrition task via the legacy dict protocol.

        Called by ``task_router.route(task, agents)``.  Merges ``task``
        into beliefs, runs the full BDI pipeline, and returns a plain dict.

        Args:
            task: Plain task dict.  May contain ``bmi``, ``conditions``,
                  ``vitals``, ``allergies``, ``prescriptions``, etc.

        Returns:
            Plain dict with keys ``"agent"``, ``"actions"``, ``"metadata"``.
        """
        self.update_beliefs(task)
        result = self.act(self.deliberate())
        return {
            "agent":   self.agent_id,
            "actions": [
                a.payload | {"action_type": a.action_type.value,
                              "urgency":     a.urgency.value}
                for a in result.actions
            ],
            "metadata": result.metadata,
        }

    # ── Private pipeline ──────────────────────────────────────────────────────

    def _build_plan(self) -> MealPlan:
        """Execute the full meal-plan selection pipeline (Listing 3).

        Steps:
        1. Compute nutrient targets.
        2. Build the excluded-tag set from conditions and allergies.
        3. For each meal slot: filter → rank → select.
        4. Local replan if caloric deviation > 300 kcal.

        Returns:
            Populated :class:`MealPlan`.
        """
        targets      = self._compute_targets()
        excluded_tags = self._build_excluded_tags()
        glucose      = self.beliefs.get("vitals", {}).get("glucose_mgdl", 100.0)

        plan = MealPlan(targets=targets)
        running_sodium = 0.0

        for slot in self._MEAL_SLOTS:
            candidates = self._candidate_meals(slot, excluded_tags)
            if not candidates:
                self._log.warning("No candidates for slot '%s' — skipping.", slot)
                continue

            remaining_sodium = targets.sodium_mg - running_sodium
            ranked = self._rank_by_fit(candidates, glucose, remaining_sodium)
            chosen = ranked[0]

            setattr(plan, slot, chosen)
            running_sodium += chosen.sodium_mg

        # Aggregate totals
        selected = [getattr(plan, s) for s in self._MEAL_SLOTS
                     if getattr(plan, s) is not None]
        plan.total_kcal      = sum(f.kcal      for f in selected)
        plan.total_sodium_mg = sum(f.sodium_mg  for f in selected)
        plan.total_fiber_g   = sum(f.fiber_g    for f in selected)

        # Local replan if caloric deviation is large
        deviation = plan.total_kcal - targets.kcal
        if abs(deviation) > 300:
            plan = self._local_replan(plan, targets, glucose, deviation)

        return plan

    def _compute_targets(self) -> NutrientTargets:
        """Derive per-patient daily targets from beliefs.

        Logic:
        * Sodium cap is read from the policy registry based on whether the
          patient has hypertension (2.3 g/day) or heart failure (1.5 g/day).
        * Calories are reduced by 200 kcal for obese patients.
        * Potassium target is restricted to 2000 mg/day for CKD patients.

        Returns:
            :class:`NutrientTargets` for this patient.
        """
        conditions = self.beliefs.get("conditions", [])
        bmi        = float(self.beliefs.get("bmi", 27.0))
        weight_kg  = self.beliefs.get("weight_kg") or (bmi * (1.70 ** 2))

        condition_key = (
            "heart_failure" if "heart_failure"  in conditions else
            "hypertension"  if "hypertension"   in conditions else
            "healthy"
        )
        sodium_cap_g = self.registry.get_sodium_cap(condition_key)

        kcal = 2000.0
        if "obesity"   in conditions:
            kcal -= 200.0
        if "underweight" in conditions:
            kcal += 200.0

        potassium_mg = 2000.0 if "ckd" in conditions else 3500.0

        return NutrientTargets(
            kcal         = kcal,
            sodium_mg    = sodium_cap_g * 1000.0,
            fiber_g      = 25.0,
            protein_g    = float(weight_kg) * 0.8,
            potassium_mg = potassium_mg,
        )

    def _build_excluded_tags(self) -> set[str]:
        """Collect food tags that must be excluded for this patient.

        Sources:
        * ``"ckd"`` condition → exclude ``"potassium_rich"`` foods.
        * Allergy flags → directly map allergen name to tag.
        * Drug-food exclusions injected by :meth:`perceive`.

        Returns:
            Set of tag strings to exclude from candidate lists.
        """
        conditions  = self.beliefs.get("conditions", [])
        allergies   = self.beliefs.get("allergies", {})
        excluded    = set(self.beliefs.get("excluded_foods", set()))

        if "ckd" in conditions:
            excluded.add("potassium_rich")
        if "dysphagia" in conditions:
            excluded.add("hard_texture")

        # Map allergy keys directly to food tags
        for allergen, active in allergies.items():
            if active:
                excluded.add(allergen)

        return excluded

    def _candidate_meals(
        self, slot: str, excluded_tags: set[str]
    ) -> list[FoodItem]:
        """Filter the food DB to valid candidates for a meal slot.

        Args:
            slot:          One of ``"breakfast"``, ``"lunch"``,
                           ``"dinner"``, ``"snack"``.
            excluded_tags: Tags that disqualify a food item.

        Returns:
            Filtered list of :class:`FoodItem` objects.  Returns all
            slot-matching items if every candidate is excluded (fail-safe).
        """
        slot_matches  = [f for f in self._food_db if f.meal_slot == slot]
        safe_items    = [
            f for f in slot_matches
            if not any(t in f.tags for t in excluded_tags)
        ]
        return safe_items if safe_items else slot_matches

    def _rank_by_fit(
        self,
        candidates: list[FoodItem],
        glucose: float,
        remaining_sodium_mg: float,
    ) -> list[FoodItem]:
        """Score and sort candidates by clinical fit (higher = better).

        Scoring heuristics:

        * If ``glucose > 140`` (post-prandial hyperglycaemia): penalise
          high glycaemic index (−GI / 100).
        * If ``remaining_sodium_mg < 600`` (near daily cap): penalise
          high-sodium items (−sodium_mg / 500).
        * Reward fibre content (+fibre_g × 0.1).
        * Reward protein content (+protein_g × 0.05).

        Args:
            candidates:          Pool of :class:`FoodItem` objects for this slot.
            glucose:             Current blood glucose (mg/dL).
            remaining_sodium_mg: Remaining sodium budget for the day.

        Returns:
            Sorted list, best candidate first.
        """
        def _score(food: FoodItem) -> float:
            s = 0.0
            if glucose > 140:
                s -= food.glycemic_index / 100.0
            if remaining_sodium_mg < 600:
                s -= food.sodium_mg / 500.0
            s += food.fiber_g   * 0.10
            s += food.protein_g * 0.05
            return s

        return sorted(candidates, key=_score, reverse=True)

    def _local_replan(
        self,
        plan: MealPlan,
        targets: NutrientTargets,
        glucose: float,
        deviation_kcal: float,
    ) -> MealPlan:
        """Adjust the snack slot to bring caloric total closer to target.

        Only modifies the snack slot to preserve meal stability.  If no
        suitable swap is available, records a note rather than forcing an
        unsatisfactory substitution.

        Args:
            plan:           Existing meal plan to adjust.
            targets:        Daily nutrient targets.
            glucose:        Current blood glucose for re-ranking.
            deviation_kcal: Positive = over target; negative = under target.

        Returns:
            Updated :class:`MealPlan` with adjusted snack and notes.
        """
        excluded_tags   = self._build_excluded_tags()
        snack_candidates = self._candidate_meals("snack", excluded_tags)

        if deviation_kcal > 0:
            # Over calories — prefer lower-calorie snack
            alternatives = sorted(snack_candidates, key=lambda f: f.kcal)
        else:
            # Under calories — prefer higher-calorie snack
            alternatives = sorted(snack_candidates, key=lambda f: f.kcal,
                                   reverse=True)

        current_snack_kcal = plan.snack.kcal if plan.snack else 0.0
        for candidate in alternatives:
            if candidate.food_id == (plan.snack.food_id if plan.snack else None):
                continue  # Skip current selection
            if abs(deviation_kcal) > abs(
                (plan.total_kcal - current_snack_kcal + candidate.kcal)
                - targets.kcal
            ):
                old_sodium  = plan.snack.sodium_mg if plan.snack else 0.0
                plan.snack  = candidate
                plan.total_kcal      = (plan.total_kcal
                                         - current_snack_kcal + candidate.kcal)
                plan.total_sodium_mg = (plan.total_sodium_mg
                                         - old_sodium + candidate.sodium_mg)
                plan.total_fiber_g   = (plan.total_fiber_g
                                         - (plan.snack.fiber_g if plan.snack
                                            else 0.0) + candidate.fiber_g)
                plan.notes.append(
                    f"Snack swapped to {candidate.name} to correct "
                    f"{deviation_kcal:+.0f} kcal deviation."
                )
                break
        else:
            plan.notes.append(
                f"Caloric deviation {deviation_kcal:+.0f} kcal — "
                "no suitable snack swap found; plan kept as-is."
            )
        return plan

    def _serialise_plan(self, plan: MealPlan) -> dict[str, Any]:
        """Convert a :class:`MealPlan` to a JSON-serialisable dict.

        Args:
            plan: The meal plan to serialise.

        Returns:
            Dict suitable for the action payload and audit log.
        """
        def _food_to_dict(f: FoodItem | None) -> dict | None:
            if f is None:
                return None
            return {
                "food_id":        f.food_id,
                "name":           f.name,
                "kcal":           f.kcal,
                "sodium_mg":      f.sodium_mg,
                "fiber_g":        f.fiber_g,
                "protein_g":      f.protein_g,
                "potassium_mg":   f.potassium_mg,
                "glycemic_index": f.glycemic_index,
                "tags":           list(f.tags),
            }

        return {
            "breakfast":       _food_to_dict(plan.breakfast),
            "lunch":           _food_to_dict(plan.lunch),
            "dinner":          _food_to_dict(plan.dinner),
            "snack":           _food_to_dict(plan.snack),
            "total_kcal":      round(plan.total_kcal, 1),
            "total_sodium_mg": round(plan.total_sodium_mg, 1),
            "total_fiber_g":   round(plan.total_fiber_g, 1),
            "targets": {
                "kcal":         plan.targets.kcal,
                "sodium_mg":    plan.targets.sodium_mg,
                "fiber_g":      plan.targets.fiber_g,
                "protein_g":    round(plan.targets.protein_g, 1),
                "potassium_mg": plan.targets.potassium_mg,
            },
            "notes": plan.notes,
        }
