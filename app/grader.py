"""
Advanced Grading System for Traffic Signal Control.

Multi-metric evaluation:
- Efficiency: Vehicles cleared per step
- Responsiveness: Emergency handling rate
- Fairness: Equitable distribution of green time
- Stability: Signal change frequency
- Congestion: Peak traffic management
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GradingMetrics:
    """Container for all grading metrics."""
    efficiency: float = 0.0
    responsiveness: float = 0.0
    fairness: float = 0.0
    stability: float = 0.0
    congestion_penalty: float = 0.0


class Grader:
    """Strict grader: 0.9+ needs near-perfection."""

    THRESHOLDS = {
        "easy": {
            "vehicles_target": 20,
            "waiting_time_max": 8,
            "efficiency_min": 1.5,
            "emergency_rate_min": 0.7
        },
        "medium": {
            "vehicles_target": 15,
            "waiting_time_max": 6,
            "efficiency_min": 2.0,
            "emergency_rate_min": 0.85
        },
        "hard": {
            "vehicles_target": 10,
            "waiting_time_max": 4,
            "efficiency_min": 2.5,
            "emergency_rate_min": 0.95
        }
    }

    def __init__(self, task: str = "easy"):
        self.task = task.lower()
        self.thresholds = self.THRESHOLDS.get(self.task, self.THRESHOLDS["easy"])

    def grade(self, state: dict, info: dict) -> tuple:
        """Calculate comprehensive score strictly between 0 and 1."""
        metrics = GradingMetrics()

        # 1. Efficiency Score (0.0 - 0.25)
        efficiency = info.get("efficiency", 0.0)
        efficiency_target = self.thresholds["efficiency_min"]
        if efficiency >= efficiency_target:
            metrics.efficiency = 0.25
        else:
            metrics.efficiency = max(0.01, (efficiency / efficiency_target) * 0.25)

        # 2. Responsiveness - Emergency handling (0.0 - 0.25)
        emergencies_total = (
            info.get("emergencies_handled", 0) +
            info.get("emergencies_missed", 0)
        )
        if emergencies_total > 0:
            response_rate = info.get("emergencies_handled", 0) / emergencies_total
            rate_target = self.thresholds["emergency_rate_min"]
            if response_rate >= rate_target:
                metrics.responsiveness = 0.25
            else:
                metrics.responsiveness = response_rate * 0.20
                if info.get("emergencies_missed", 0) > 2:
                    metrics.responsiveness *= 0.5
        else:
            metrics.responsiveness = 0.20

        # 3. Fairness - Equal green time distribution (0.0 - 0.15)
        total_vehicles = state.get("total_vehicles", 0)
        lanes = state.get("lanes", {})

        if lanes:
            vehicle_counts = [lane.get("vehicles", 0) for lane in lanes.values()]
            if max(vehicle_counts) > 0:
                variance = sum(
                    (v - sum(vehicle_counts) / len(vehicle_counts)) ** 2
                    for v in vehicle_counts
                ) / len(vehicle_counts)
                fairness_score = max(0.0, 0.15 - (variance / 100))
                metrics.fairness = min(0.15, fairness_score)
            else:
                metrics.fairness = 0.15

        # 4. Stability - Signal change frequency (0.0 - 0.10)
        signal_changes = info.get("signal_changes", 0)
        steps = state.get("steps", 1)
        change_rate = signal_changes / max(steps, 1)

        if change_rate < 0.3:
            metrics.stability = 0.10
        elif change_rate < 0.5:
            metrics.stability = 0.05
        else:
            metrics.stability = 0.0

        # 5. Congestion Penalty (-0.30 to 0.0)
        avg_waiting = info.get("avg_waiting_time", 0)
        waiting_max = self.thresholds["waiting_time_max"]

        if avg_waiting > waiting_max * 2:
            metrics.congestion_penalty = -0.10
        elif avg_waiting > waiting_max:
            metrics.congestion_penalty = -0.15 * (avg_waiting - waiting_max) / waiting_max

        # Additional penalties
        extra_penalties = 0.0

        if total_vehicles > 100:
            extra_penalties -= 0.15
        elif total_vehicles > 80:
            extra_penalties -= 0.10

        if info.get("emergencies_missed", 0) > 3:
            extra_penalties -= 0.10

        congestion_events = info.get("congestion_events", 0)
        if congestion_events > 5:
            extra_penalties -= min(0.15, congestion_events * 0.03)

        # Calculate final score
        raw_score = (
            metrics.efficiency +
            metrics.responsiveness +
            metrics.fairness +
            metrics.stability +
            metrics.congestion_penalty +
            extra_penalties
        )

        # Apply difficulty scaling
        difficulty_scaling = {
            "easy": 1.0,
            "medium": 0.85,
            "hard": 0.70
        }
        scaling = difficulty_scaling.get(self.task, 1.0)

        # Strictly between 0 and 1 — never exactly 0.0 or 1.0
        final_score = max(0.01, min(0.99, raw_score * scaling))

        metrics_dict = {
            "efficiency": round(metrics.efficiency, 3),
            "responsiveness": round(metrics.responsiveness, 3),
            "fairness": round(metrics.fairness, 3),
            "stability": round(metrics.stability, 3),
            "congestion_penalty": round(metrics.congestion_penalty, 3),
            "extra_penalties": round(extra_penalties, 3),
            "difficulty_scaling": scaling
        }

        return round(final_score, 4), metrics_dict


def grade(state: dict, task: str = "easy", info: dict = None) -> tuple:
    """Convenience function for grading."""
    info = info or {}
    grader = Grader(task)
    return grader.grade(state, info)