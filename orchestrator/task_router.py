"""
task_router.py
==============
Routes resolved tasks to the appropriate specialised agent.
"""


class TaskRouter:
    """Maps task types to agents."""

    TASK_AGENT_MAP = {
        "medication_reminder": "medicine_agent",
        "medication_schedule": "medicine_agent",
        "escalate_drug_safety": "medicine_agent",
        "dietary_modification": "nutrition_agent",
        "meal_plan": "nutrition_agent",
        "sodium_advisory": "nutrition_agent",
        "lifestyle_prompt": "lifestyle_agent",
        "sleep_adjustment": "lifestyle_agent",
        "walk_prompt": "lifestyle_agent",
        "escalate": "emergency_agent",
        "escalate_to_clinician": "emergency_agent",
        "repeat_measurement": "emergency_agent",
    }

    def route(self, task: dict, agents: dict):
        """
        Route a single task to its agent and execute.

        Parameters
        ----------
        task   : dict — task to execute
        agents : dict — {agent_id: agent_instance}

        Returns
        -------
        dict — result from agent.execute()
        """
        task_type = task.get("type", "unknown")
        agent_id = self.TASK_AGENT_MAP.get(task_type, "emergency_agent")
        agent = agents.get(agent_id)
        if agent is None:
            return {"error": f"No agent found for task type: {task_type}"}
        return agent.execute(task)
