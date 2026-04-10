"""FastAPI application implementing OpenEnv spec."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.env import TrafficEnv
from app.models import (
    Observation, Action, StepResponse,
    ResetResponse, GraderResponse, EpisodeInfo, LaneState
)
from app.grader import grade

app = FastAPI(
    title="Advanced Traffic Signal Control Environment",
    description="Multi-directional traffic optimization with emergencies, weather, and dynamic patterns",
    version="2.0.0"
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
        "message": "Advanced Traffic Optimization Environment",
        "status": "running",
        "version": "2.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grader", "/tasks", "/metrics"]
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

    # Smart emergency override
    if action_str == "RED":
        # Check for emergencies - auto-prioritize if present
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
                "description": "Basic traffic management with low emergency frequency",
                "target_score": 0.75,
                "features": ["4-directional traffic", "basic emergencies", "clear weather"]
            },
            {
                "name": "medium",
                "description": "Traffic + frequent emergencies + rush hour patterns",
                "target_score": 0.80,
                "features": ["increased emergency rate", "rush hour multiplier", "traffic spikes"]
            },
            {
                "name": "hard",
                "description": "Full complexity: weather, spikes, emergencies, gridlock risk",
                "target_score": 0.85,
                "features": ["weather changes", "high emergency rate", "frequent spikes", "strict grading"]
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
            "active_signal": env.active_direction.value if env.active_direction else None
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
        "recommended_action": f"GREEN_{best.value.upper()}"
    }


def _parse_observation(state: dict) -> Observation:
    """Parse state dict into Observation model."""
    lanes = {}
    for key, value in state.get("lanes", {}).items():
        lanes[key] = LaneState(**value)

    return Observation(
        lanes=lanes,
        active_signal=state.get("active_signal"),
        signal_state=state.get("signal_state", "RED"),
        weather=state.get("weather", "clear"),
        time_of_day=state.get("time_of_day", 12),
        total_vehicles=state.get("total_vehicles", 0),
        steps=state.get("steps", 0),
        task=state.get("task", "easy")
    )