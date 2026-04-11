"""FastAPI server for traffic signal control environment."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.env import TrafficEnv
from app.models import (
    Observation, Action, StepResponse,
    ResetResponse, GraderResponse, EpisodeInfo, LaneState, UpstreamHandoff
)
from app.grader import grade

app = FastAPI(
    title="DualEye H-MARL Traffic Signal Control",
    description="Bangalore-specific multi-intersection traffic optimization with emergency prioritization, accident mode, and upstream handoff",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env: TrafficEnv = None
current_task: str = "easy"


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "DualEye H-MARL Traffic Optimization Environment",
        "status": "running",
        "version": "3.0.0",
        "location": "Bangalore, India",
        "endpoints": ["/reset", "/step", "/state", "/grader", "/tasks", "/metrics", "/priority"]
    }


@app.api_route("/reset", methods=["GET", "POST"], response_model=ResetResponse)
def reset(task: str = "easy", seed: int = None):
    """Reset environment to initial state."""
    global env, current_task
    current_task = task
    env = TrafficEnv(task=task, seed=seed)
    state = env.reset()
    return ResetResponse(observation=_parse_observation(state))


@app.get("/state")
def get_state():
    """Get current environment state."""
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {"observation": _parse_observation(env.state()).model_dump()}


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """Execute action and return observation, reward, done."""
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action_str = action.action.upper()

    if action_str == "RED":
        for lane in env.lanes.values():
            if lane.emergency:
                action_str = "PRIORITY_GREEN"
                break

    state, reward, done, info = env.step(action_str)

    return StepResponse(
        observation=_parse_observation(state),
        reward=reward,
        done=done,
        info=EpisodeInfo(**info) if info else None
    )


@app.get("/grader", response_model=GraderResponse)
def get_score(difficulty: str = "easy"):
    """Get comprehensive performance score."""
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    state = env.state()
    info = env._get_info()
    score, metrics = grade(state, difficulty, info)

    return GraderResponse(
        score=score,
        task=difficulty,
        metrics=metrics
    )


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions."""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Basic Bangalore traffic with low emergency frequency",
                "target_score": 0.75,
                "features": [
                    "4-directional traffic",
                    "basic emergencies",
                    "clear weather",
                    "Bangalore rush hour patterns"
                ]
            },
            {
                "name": "medium",
                "description": "Rush hour + emergencies + accidents",
                "target_score": 0.80,
                "features": [
                    "increased emergency rate",
                    "Silk Board morning rush",
                    "KR Puram evening rush",
                    "accident mode"
                ]
            },
            {
                "name": "hard",
                "description": "Full Bangalore chaos: weather, accidents, gridlock risk",
                "target_score": 0.85,
                "features": [
                    "dynamic weather",
                    "high emergency rate",
                    "frequent accidents",
                    "upstream handoff",
                    "gridlock risk"
                ]
            }
        ]
    }


@app.get("/metrics")
def get_metrics():
    """Get current episode metrics."""
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    return {
        "episode_stats": env._get_info(),
        "current_state": {
            "total_vehicles": env.get_total_vehicles(),
            "avg_waiting_time": env._get_average_waiting_time(),
            "active_signal": env.active_direction.value if env.active_direction else None,
            "time_of_day": env.time_of_day,
            "weather": env.weather.value
        }
    }


@app.get("/priority")
def get_priority_lanes():
    """Get lane priority scores for decision support."""
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    priorities = {}
    for direction, lane in env.lanes.items():
        priorities[direction.value] = {
            "score": lane.get_priority_score(),
            "vehicles": lane.vehicles,
            "emergency": lane.emergency,
            "waiting_time": lane.waiting_time
        }

    best = env._get_highest_priority_lane()
    return {
        "priorities": priorities,
        "recommended_action": f"GREEN_{best.value.upper()}",
        "upstream_handoff": env.state().get("upstream_handoff", {})
    }


def _parse_observation(state: dict) -> Observation:
    """Convert raw state dict to Observation model."""
    lanes = {}
    for key, value in state.get("lanes", {}).items():
        lanes[key] = LaneState(**value)

    handoff_data = state.get("upstream_handoff")
    handoff = UpstreamHandoff(**handoff_data) if handoff_data else None

    return Observation(
        lanes=lanes,
        active_signal=state.get("active_signal"),
        signal_state=state.get("signal_state", "RED"),
        weather=state.get("weather", "clear"),
        time_of_day=state.get("time_of_day", 12),
        total_vehicles=state.get("total_vehicles", 0),
        steps=state.get("steps", 0),
        task=state.get("task", "easy"),
        upstream_handoff=handoff
    )
