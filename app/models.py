"""Pydantic models for OpenEnv spec compliance."""

from pydantic import BaseModel
from typing import Literal, Optional, Dict


class LaneState(BaseModel):
    """State of a single lane."""
    vehicles: int
    emergency: bool
    waiting_time: int


class UpstreamHandoff(BaseModel):
    """Upstream handoff message between intersections."""
    incoming_direction: str
    vehicle_count: int
    eta_seconds: int
    emergency_incoming: bool
    weather: str
    time_of_day: int


class Observation(BaseModel):
    """Full environment observation."""
    lanes: Dict[str, LaneState]
    active_signal: Optional[str]
    signal_state: str
    weather: str
    time_of_day: int
    total_vehicles: int
    steps: int
    task: str
    upstream_handoff: Optional[UpstreamHandoff] = None


class Action(BaseModel):
    """Agent action - which signal to activate."""
    action: Literal[
        "GREEN_NORTH",
        "GREEN_SOUTH",
        "GREEN_EAST",
        "GREEN_WEST",
        "PRIORITY_GREEN",
        "RED",
        "HOLD"
    ]


class EpisodeInfo(BaseModel):
    """Episode statistics."""
    vehicles_cleared: int = 0
    emergencies_handled: int = 0
    emergencies_missed: int = 0
    congestion_events: int = 0
    accident_events: int = 0
    avg_waiting_time: float = 0.0
    signal_changes: int = 0
    efficiency: float = 0.0


class StepResponse(BaseModel):
    """Response from step() endpoint."""
    observation: Observation
    reward: float
    done: bool
    info: Optional[EpisodeInfo] = None


class ResetResponse(BaseModel):
    """Response from reset() endpoint."""
    observation: Observation


class GraderResponse(BaseModel):
    """Response from grader endpoint."""
    score: float
    task: str
    metrics: Dict[str, float]
