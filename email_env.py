from pydantic import BaseModel
from typing import Dict, Any


class Observation(BaseModel):
    ticket_text: str
    current_stage: str
    step_count: int


class Action(BaseModel):
    action_type: str
    resolution_note: str


class Reward(BaseModel):
    score: float
    message: str


class SupportAgentEnv:
    def __init__(self):
        self.task = None
        self.steps = 0
        self.current_stage = "analysis"

    def reset(self, task_id=1):
        self.steps = 0
        self.current_stage = "analysis"

        tasks = {
            1: {
                "task_name": "refund_issue",
                "ticket_text": "Customer reports payment deducted twice and requests refund.",
                "expected_flow": [
                    "analyze_ticket",
                    "process_refund",
                    "close_ticket"
                ]
            },
            2: {
                "task_name": "technical_issue",
                "ticket_text": "User app crashes repeatedly after login.",
                "expected_flow": [
                    "analyze_ticket",
                    "escalate_engineering",
                    "close_ticket"
                ]
            },
            3: {
                "task_name": "priority_enterprise",
                "ticket_text": "Enterprise client system outage impacting 500 users.",
                "expected_flow": [
                    "analyze_ticket",
                    "mark_critical",
                    "escalate_management"
                ]
            }
        }

        self.task = tasks[task_id]

        return Observation(
            ticket_text=self.task["ticket_text"],
            current_stage=self.current_stage,
            step_count=self.steps
        )

    def step(self, action: Action):
        self.steps += 1

        expected_action = self.task["expected_flow"][self.steps - 1]

        score = 0.0
        done = False

        if action.action_type == expected_action:
            score += 0.6

        score += min(0.1 * self.steps, 0.4)

        if self.steps >= 3:
            done = True

        score = min(score, 1.0)

        stages = ["analysis", "processing", "resolution"]

        self.current_stage = stages[min(self.steps, 2)]

        return (
            Observation(
                ticket_text=self.task["ticket_text"],
                current_stage=self.current_stage,
                step_count=self.steps
            ),
            Reward(
                score=score,
                message="Workflow step evaluated"
            ),
            done,
            {
                "task_name": self.task["task_name"],
                "workflow_progress": self.steps / 3
            }
        )

    def state(self):
        return {
            "task": self.task,
            "steps": self.steps,
            "stage": self.current_stage
        }