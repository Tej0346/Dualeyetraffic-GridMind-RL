"""OpenEnv baseline inference script for traffic signal control."""

import os
import asyncio
from typing import List, Dict, Any
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

ENV_URL = os.getenv("ENV_URL", "https://tejas01720-gridmind-rl.hf.space")

MAX_STEPS = 50
MAX_TOTAL_REWARD = 100.0
SUCCESS_SCORE_THRESHOLD = 0.7

TASK_NAME = "traffic-signal-control"
BENCHMARK = "openenv-traffic-signal-v1"

TEMPERATURE = 0.2
MAX_TOKENS = 20

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def load_environment():
    """Load the OpenEnv environment."""
    return {"url": ENV_URL}


async def reset_env(env, task: str = "easy") -> Any:
    """Reset environment and get initial observation."""
    import requests
    response = requests.post(f"{env['url']}/reset", params={"task": task})
    return response.json()


async def step_env(env, action: str) -> Any:
    """Execute action and get result."""
    import requests
    response = requests.post(f"{env['url']}/step", json={"action": action})
    return response.json()


def get_model_action(state: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Get action from LLM with state context."""
    prompt = f"""You are a traffic signal controller AI. Control a 4-way intersection.

Total vehicles: {analysis.get('total_vehicles', 0)}
Weather: {state.get('weather', 'clear')}
Time: {state.get('time_of_day', 12)}:00
Active signal: {state.get('active_signal', 'none')}

Lane Status:
"""
    for direction, lane in state.get("lanes", {}).items():
        emergency_marker = " [EMERGENCY]" if lane.get("emergency") else ""
        prompt += f"- {direction.upper()}: {lane.get('vehicles', 0)} vehicles, waiting {lane.get('waiting_time', 0)}s{emergency_marker}\n"

    prompt += "\nChoose ONE action: GREEN_NORTH, GREEN_SOUTH, GREEN_EAST, GREEN_WEST, PRIORITY_GREEN, HOLD, or RED\n"
    prompt += "Reply with ONLY the action name."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a traffic signal controller. Choose the best action to minimize congestion."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        action = response.choices[0].message.content.strip().upper()

        valid_actions = [
            "GREEN_NORTH", "GREEN_SOUTH", "GREEN_EAST", "GREEN_WEST",
            "PRIORITY_GREEN", "HOLD", "RED"
        ]

        if action in valid_actions:
            return action

        for valid in valid_actions:
            if valid in action or action in valid:
                return valid

        return "GREEN_NORTH"

    except Exception as exc:
        return smart_heuristic(state, analysis)


def smart_heuristic(state: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Rule-based fallback when LLM is unavailable."""
    priorities = analysis.get("priorities", {})
    emergencies = analysis.get("emergencies", [])

    if emergencies:
        return "PRIORITY_GREEN"

    if priorities:
        best = max(priorities, key=priorities.get)
        return f"GREEN_{best.upper()}"

    return "GREEN_NORTH"


def analyze_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze current state and compute recommendations."""
    lanes = state.get("lanes", {})
    total_vehicles = state.get("total_vehicles", 0)

    priorities = {}
    emergencies = []

    for direction, lane in lanes.items():
        vehicles = lane.get("vehicles", 0)
        emergency = lane.get("emergency", False)
        waiting = lane.get("waiting_time", 0)

        priority = vehicles + waiting * 0.5 + (50 if emergency else 0)
        priorities[direction] = priority

        if emergency:
            emergencies.append(direction)

    return {
        "priorities": priorities,
        "emergencies": emergencies,
        "total_vehicles": total_vehicles,
        "congestion_level": "high" if total_vehicles > 60 else "medium" if total_vehicles > 30 else "low"
    }


async def run_episode(env, task: str) -> Dict[str, Any]:
    """Run single episode and return results."""
    result = await reset_env(env, task)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            obs = result.get("observation", {})
            done = result.get("done", False)

            if done:
                break

            analysis = analyze_state(obs)
            action = get_model_action(obs, analysis)

            result = await step_env(env, action)
            step_reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(step_reward)
            steps_taken = step

            log_step(step=step, action=action, reward=step_reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
        "total_reward": sum(rewards)
    }


async def main() -> None:
    """Run evaluation on all tasks."""
    env = await load_environment()

    results = []

    for task in ["easy", "medium", "hard"]:
        result = await run_episode(env, task)
        results.append(result)
        await asyncio.sleep(0.5)

    avg = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage Score: {avg:.4f}")

    return results


if __name__ == "__main__":
    asyncio.run(main())