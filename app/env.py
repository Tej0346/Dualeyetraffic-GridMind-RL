import random
from typing import Tuple, Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class Weather(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"


class SignalState(Enum):
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


@dataclass
class Lane:
    """Single lane with vehicles and properties."""
    vehicles: int = 0
    emergency: bool = False
    waiting_time: int = 0

    def get_priority_score(self) -> float:
        base = self.vehicles * 1.0
        waiting_penalty = self.waiting_time * 0.5
        emergency_bonus = 50.0 if self.emergency else 0.0
        return base + waiting_penalty + emergency_bonus


class TrafficEnv:
    """4-way intersection with Bangalore-specific traffic patterns."""

    BASE_FLOW_RATE = {
        Weather.CLEAR: 4,
        Weather.RAIN: 3,
        Weather.FOG: 2,
        Weather.SNOW: 1
    }

    # Bangalore-specific rush hour patterns
    BANGALORE_PATTERNS = {
        "morning_rush": (7, 10),   # Silk Board nightmare
        "evening_rush": (17, 20),  # KR Puram chaos
        "lunch_lull": (12, 14),    # Relatively calm
        "night_mode": (22, 6)      # Empty roads
    }

    BANGALORE_INTERSECTIONS = {
        "silk_board": {"peak_multiplier": 3.5, "base_vehicles": 25},
        "kr_puram": {"peak_multiplier": 2.8, "base_vehicles": 20},
        "hebbal": {"peak_multiplier": 2.5, "base_vehicles": 18}
    }

    RUSH_HOUR_MULTIPLIER = 2.5

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        self.task = task
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self.lanes: Dict[Direction, Lane] = {
            Direction.NORTH: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.SOUTH: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.EAST: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.WEST: Lane(vehicles=self.rng.randint(8, 15)),
        }

        self.active_direction: Optional[Direction] = None
        self.signal_state: SignalState = SignalState.RED

        self.weather = self.rng.choice(list(Weather))
        self.time_of_day = self.rng.randint(6, 22)
        self.steps = 0
        self.max_steps = 50

        self.total_vehicles_cleared = 0
        self.total_waiting_time = 0
        self.emergencies_handled = 0
        self.emergencies_missed = 0
        self.signal_changes = 0
        self.congestion_events = 0
        self.accident_events = 0

        self.reward_history: List[float] = []

        return self.state()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return (observation, reward, done, info)."""
        self.steps += 1
        prev_total_vehicles = self.get_total_vehicles()

        reward = self._execute_action(action)

        self._simulate_traffic_flow()
        self._generate_new_traffic()
        self._update_waiting_times()

        self._spawn_emergencies()
        self._spawn_accident()
        self._maybe_change_weather()

        reward += self._calculate_step_reward(prev_total_vehicles, action)

        self.reward_history.append(reward)

        done = self.steps >= self.max_steps or self._is_gridlocked()

        info = self._get_info()

        reward = reward / 10.0
        return self.state(), reward, done, info

    def _execute_action(self, action: str) -> float:
        """Execute the chosen action and return immediate reward."""
        reward = 0.0

        if action == "RED":
            self.signal_state = SignalState.RED
            self.active_direction = None
            reward -= 2.0

        elif action == "HOLD":
            if self.active_direction is not None:
                reward += 0.5
            else:
                reward -= 1.0

        elif action.startswith("GREEN_"):
            direction_str = action.replace("GREEN_", "").lower()
            try:
                direction = Direction(direction_str)
                reward += self._set_green_signal(direction)
            except ValueError:
                reward -= 5.0

        elif action == "PRIORITY_GREEN":
            priority_lane = self._get_highest_priority_lane()
            reward += self._set_green_signal(priority_lane)

        return reward

    def _set_green_signal(self, direction: Direction) -> float:
        """Set GREEN signal for a direction and process vehicles."""
        reward = 0.0

        if self.active_direction != direction:
            self.signal_changes += 1
            reward -= 0.5
            if self.steps > 1 and self.signal_changes > self.steps * 0.3:
                reward -= 1.0

        self.active_direction = direction
        self.signal_state = SignalState.GREEN

        lane = self.lanes[direction]
        flow_rate = self.BASE_FLOW_RATE.get(self.weather, 4)

        if lane.emergency:
            flow_rate = min(lane.vehicles, flow_rate + 3)
            self.emergencies_handled += 1
            reward += 15.0
            lane.emergency = False

        vehicles_cleared = min(lane.vehicles, flow_rate)
        lane.vehicles -= vehicles_cleared
        self.total_vehicles_cleared += vehicles_cleared

        reward += vehicles_cleared * 1.5

        if lane.waiting_time > 5:
            reward += vehicles_cleared * 0.5

        return reward

    def _simulate_traffic_flow(self) -> None:
        """Vehicles in non-active lanes wait."""
        for direction, lane in self.lanes.items():
            if direction != self.active_direction:
                lane.waiting_time += 1

    def _generate_new_traffic(self) -> None:
        """Generate new vehicles using Bangalore-specific patterns."""
        is_morning_rush = 7 <= self.time_of_day <= 10
        is_evening_rush = 17 <= self.time_of_day <= 20
        is_lunch = 12 <= self.time_of_day <= 14

        if is_morning_rush:
            multiplier = 3.5  # Silk Board morning peak
        elif is_evening_rush:
            multiplier = 3.0  # KR Puram evening peak
        elif is_lunch:
            multiplier = 1.5  # Lunch hour moderate traffic
        else:
            multiplier = 1.0

        if self.task in ["medium", "hard"]:
            spike_chance = 0.15 if self.task == "hard" else 0.1
            if self.rng.random() < spike_chance:
                multiplier *= 2.0

        for direction, lane in self.lanes.items():
            base_arrival = self.rng.randint(1, 3)
            arrivals = int(base_arrival * multiplier)

            if self.weather in [Weather.RAIN, Weather.SNOW]:
                arrivals += self.rng.randint(0, 2)

            lane.vehicles += arrivals
            lane.vehicles = min(lane.vehicles, 50)

        self.time_of_day = (self.time_of_day + 1) % 24

    def _update_waiting_times(self) -> None:
        """Track waiting times."""
        for lane in self.lanes.values():
            self.total_waiting_time += lane.vehicles * 0.1

    def _spawn_emergencies(self) -> None:
        """Randomly spawn emergency vehicles."""
        emergency_chance = {
            "easy": 0.05,
            "medium": 0.1,
            "hard": 0.15
        }.get(self.task, 0.1)

        if self.rng.random() < emergency_chance:
            direction = self.rng.choice(list(Direction))
            self.lanes[direction].emergency = True

    def _spawn_accident(self) -> None:
        """Random accident blocks a lane temporarily — Bangalore style."""
        accident_chance = {
            "easy": 0.02,
            "medium": 0.05,
            "hard": 0.08
        }.get(self.task, 0.02)

        if self.rng.random() < accident_chance:
            direction = self.rng.choice(list(Direction))
            self.lanes[direction].vehicles = min(
                self.lanes[direction].vehicles + 15, 50
            )
            self.lanes[direction].waiting_time += 5
            self.accident_events += 1

    def _maybe_change_weather(self) -> None:
        """Weather can change during episode (hard mode)."""
        if self.task == "hard" and self.rng.random() < 0.05:
            self.weather = self.rng.choice(list(Weather))

    def _get_highest_priority_lane(self) -> Direction:
        """Get the lane with highest priority score."""
        best_direction = Direction.NORTH
        best_score = -float('inf')

        for direction, lane in self.lanes.items():
            score = lane.get_priority_score()
            if score > best_score:
                best_score = score
                best_direction = direction

        return best_direction

    def _calculate_step_reward(self, prev_vehicles: int, action: str) -> float:
        """Calculate comprehensive step reward."""
        reward = 0.0
        current_vehicles = self.get_total_vehicles()

        vehicle_delta = prev_vehicles - current_vehicles
        reward += vehicle_delta * 2.0

        if current_vehicles > 80:
            reward -= 5.0
            self.congestion_events += 1
        elif current_vehicles > 60:
            reward -= 3.0
        elif current_vehicles > 40:
            reward -= 1.0

        avg_waiting = self._get_average_waiting_time()
        if avg_waiting > 10:
            reward -= (avg_waiting - 10) * 0.5

        for lane in self.lanes.values():
            if lane.emergency and lane.waiting_time > 3:
                reward -= 5.0
                self.emergencies_missed += 1

        efficiency = self.total_vehicles_cleared / max(self.steps, 1)
        if efficiency > 2.0:
            reward += 1.0

        return reward

    def _is_gridlocked(self) -> bool:
        """Check if intersection is gridlocked."""
        total_vehicles = self.get_total_vehicles()
        if total_vehicles > 150:
            return True

        gridlocked_lanes = sum(
            1 for lane in self.lanes.values() if lane.vehicles > 30
        )
        return gridlocked_lanes >= 3

    def _get_average_waiting_time(self) -> float:
        """Calculate average waiting time across all lanes."""
        total = sum(lane.waiting_time for lane in self.lanes.values())
        return total / len(self.lanes)

    def get_total_vehicles(self) -> int:
        """Get total vehicles across all lanes."""
        return sum(lane.vehicles for lane in self.lanes.values())

    def _get_info(self) -> Dict[str, Any]:
        """Compile episode statistics."""
        return {
            "vehicles_cleared": self.total_vehicles_cleared,
            "emergencies_handled": self.emergencies_handled,
            "emergencies_missed": self.emergencies_missed,
            "congestion_events": self.congestion_events,
            "accident_events": self.accident_events,
            "avg_waiting_time": self._get_average_waiting_time(),
            "signal_changes": self.signal_changes,
            "efficiency": self.total_vehicles_cleared / max(self.steps, 1)
        }

    def state(self) -> Dict[str, Any]:
        """Return current observation with upstream handoff."""
        lane_states = {}
        for direction in Direction:
            lane = self.lanes[direction]
            lane_states[direction.value] = {
                "vehicles": lane.vehicles,
                "emergency": lane.emergency,
                "waiting_time": lane.waiting_time
            }

        # Upstream handoff message — DualEye innovation
        highest = self._get_highest_priority_lane()
        handoff = {
            "incoming_direction": highest.value,
            "vehicle_count": self.lanes[highest].vehicles,
            "eta_seconds": max(5, 30 - self.lanes[highest].waiting_time),
            "emergency_incoming": self.lanes[highest].emergency,
            "weather": self.weather.value,
            "time_of_day": self.time_of_day
        }

        return {
            "lanes": lane_states,
            "active_signal": self.active_direction.value if self.active_direction else None,
            "signal_state": self.signal_state.value,
            "weather": self.weather.value,
            "time_of_day": self.time_of_day,
            "total_vehicles": self.get_total_vehicles(),
            "steps": self.steps,
            "task": self.task,
            "upstream_handoff": handoff
        }
