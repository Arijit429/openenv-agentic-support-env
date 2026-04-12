from pydantic import BaseModel
from typing import List


class Meeting(BaseModel):
    meeting_id: str
    owner: str
    time: str
    priority: int


class Observation(BaseModel):
    calendar: List[Meeting]
    incident_level: str
    participants_required: List[str]
    step_count: int


class Action(BaseModel):
    action_type: str
    target_meeting_id: str
    new_time: str
    reason: str


class Reward(BaseModel):
    score: float
    message: str


class CalendarOrchestrationEnv:
    def __init__(self):
        self.steps = 0
        self.state_data = None

        self.tasks = {
            1: {
                "incident_level": "low",
                "participants_required": ["CTO"]
            },
            2: {
                "incident_level": "medium",
                "participants_required": ["CTO", "DevOps"]
            },
            3: {
                "incident_level": "critical",
                "participants_required": ["CTO", "DevOps", "ClientManager"]
            },
            4: {
                "incident_level": "timezone_conflict",
                "participants_required": ["Bangalore", "London", "San Francisco"]
            },
            5: {
                "incident_level": "optimization",
                "participants_required": ["All"]
            }
        }

    def reset(self, task_id=1):
        self.steps = 0

        task = self.tasks[task_id]

        self.state_data = {
            "calendar": [
                Meeting(
                    meeting_id="m1",
                    owner="CTO",
                    time="15:00",
                    priority=3
                ),
                Meeting(
                    meeting_id="m2",
                    owner="DevOps",
                    time="15:00",
                    priority=2
                )
            ],
            "incident_level": task["incident_level"],
            "participants_required": task["participants_required"]
        }

        return Observation(
            calendar=self.state_data["calendar"],
            incident_level=self.state_data["incident_level"],
            participants_required=self.state_data["participants_required"],
            step_count=self.steps
        )

    def step(self, action: Action):
        self.steps += 1

        score = 0.0

        conflict_removed = action.new_time != "15:00"
        priority_respected = action.reason in [
            "critical_incident",
            "optimization",
            "timezone_resolution"
        ]
        minimal_disruption = action.target_meeting_id == "m2"
        participants_available = len(
            self.state_data["participants_required"]
        ) > 0
        urgency_handled = (
            self.state_data["incident_level"] != "low"
        )

        if conflict_removed:
            score += 0.25

        if priority_respected:
            score += 0.25

        if minimal_disruption:
            score += 0.20

        if participants_available:
            score += 0.15

        if urgency_handled:
            score += 0.15

        safe_score = max(0.01, min(round(score, 2), 0.99))

        done = self.steps >= 3

        return (
            Observation(
                calendar=self.state_data["calendar"],
                incident_level=self.state_data["incident_level"],
                participants_required=self.state_data["participants_required"],
                step_count=self.steps
            ),
            Reward(
                score=safe_score,
                message="Advanced benchmark step evaluated"
            ),
            done,
            {
                "workflow_progress": self.steps / 3
            }
        )

    def state(self):
        return self.state_data