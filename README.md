---
title: GridMind-RL
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - bangalore
---

# DualEye H-MARL Traffic Signal Control

A Bangalore-specific research-grade OpenEnv environment for training and evaluating AI agents on real-world traffic signal optimization at iconic junctions like Silk Board, KR Puram, and Hebbal.

## Why This Environment?

Traffic signal control affects millions daily in Bangalore. This environment simulates real urban intersection management with Bangalore-specific rush hour patterns, upstream handoff messages between intersections, accident mode, emergency green wave, and dynamic weather conditions.

## Actions

- GREEN_NORTH — Activate north signal
- GREEN_SOUTH — Activate south signal
- GREEN_EAST — Activate east signal
- GREEN_WEST — Activate west signal
- PRIORITY_GREEN — Auto-clear highest priority lane
- HOLD — Keep current signal
- RED — All signals red

## Observations

Each step returns lanes (vehicles, emergency, waiting_time for each direction), active_signal, signal_state, weather, time_of_day, total_vehicles, steps, and upstream_handoff (incoming_direction, vehicle_count, eta_seconds, emergency_incoming).

## Tasks

Easy — Basic traffic with low emergency frequency. Target score 0.75.

Medium — Silk Board rush hour, accidents, higher emergencies. Target score 0.80.

Hard — Weather changes, frequent accidents, gridlock risk. Target score 0.85.

## Bangalore Traffic Patterns

Morning rush 7-10 AM at Silk Board — 3.5x multiplier.
Lunch lull 12-2 PM — 1.5x multiplier.
Evening rush 5-8 PM at KR Puram — 3.0x multiplier.
Night mode 10 PM-6 AM — 1.0x multiplier.

## Reward Function

Positive rewards include vehicle clearance (+1.5 per vehicle), emergency handling (+15.0), and stability bonus (+0.5). Penalties include missed emergency (-5.0), high congestion (-3.0), excessive signal changes (-1.0), and accident spikes.

## API Endpoints

- GET/POST /reset?task=easy — Initialize environment
- POST /step — Execute action
- GET /state — Current state
- GET /grader?difficulty=easy — Performance score
- GET /tasks — Available tasks
- GET /metrics — Episode statistics
- GET /priority — Lane priority scores

## Setup

```bash
pip install -r requirements.txt
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token"
docker build -t traffic-env .
docker run -p 7860:7860 traffic-env
python inference.py
Baseline Scores
Easy — 0.65 to 0.75. Basic traffic handling.
Medium — 0.55 to 0.65. Rush hour challenges.
Hard — 0.45 to 0.55. Weather and accidents.
Project Structure
app/env.py — Main environment logic
app/grader.py — Scoring system
app/main.py — FastAPI server
app/models.py — Pydantic models
server/app.py — Server entry point
inference.py — Baseline inference script
openenv.yaml — OpenEnv specification
Dockerfile — Container config
requirements.txt — Python dependencies
License
MIT
