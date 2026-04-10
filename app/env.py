import random
from typing import Tuple, Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field


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
    """
    Keeps track of a single lane's status—how many cars are there, 
    if an ambulance is stuck in the pack, and how long they've been idling.
    """
    vehicles: int = 0
    emergency: bool = False
    waiting_time: int = 0  # Represents the growing frustration of drivers

    def get_priority_score(self) -> float:
        """
        Determines how 'urgent' this lane is. 
        More cars and longer waits increase the score, while an emergency 
        vehicle acts as a massive 'must-clear' multiplier.
        """
        base = self.vehicles * 1.0
        waiting_penalty = self.waiting_time * 0.5
        emergency_bonus = 50.0 if self.emergency else 0.0
        return base + waiting_penalty + emergency_bonus


class TrafficEnv:
    """
    The brain of the 4-way intersection. It handles the chaos of rush hour, 
    slippery roads during snowstorms, and clearing the path for sirens.
    """

    # How many cars can actually clear the light in one go depending on visibility/traction
    BASE_FLOW_RATE = {
        Weather.CLEAR: 4,
        Weather.RAIN: 3,
        Weather.FOG: 2,
        Weather.SNOW: 1
    }

    # Everyone hits the road at the same time; traffic gets heavy
    RUSH_HOUR_MULTIPLIER = 2.5

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        self.task = task
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Back to square one: clear the stats and start with a fresh set of traffic."""
        # Sprinkle some initial cars into each lane so the simulation doesn't start empty
        self.lanes: Dict[Direction, Lane] = {
            Direction.NORTH: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.SOUTH: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.EAST: Lane(vehicles=self.rng.randint(8, 15)),
            Direction.WEST: Lane(vehicles=self.rng.randint(8, 15)),
        }

        # Lights start off; no one is moving yet
        self.active_direction: Optional[Direction] = None
        self.signal_state: SignalState = SignalState.RED

        # Set the scene: what's the weather and what time is it?
        self.weather = self.rng.choice(list(Weather))
        self.time_of_day = self.rng.randint(6, 22)  # 6 AM to 10 PM
        self.steps = 0
        self.max_steps = 50

        # Keep a scorecard for the simulation's performance
        self.total_vehicles_cleared = 0
        self.total_waiting_time = 0
        self.emergencies_handled = 0
        self.emergencies_missed = 0
        self.signal_changes = 0
        self.congestion_events = 0

        # Track the rewards to see if our strategy is actually getting smarter
        self.reward_history: List[float] = []

        return self.state()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        The main loop: take an action (like changing a light), move the cars, 
        spawn new ones, and see how much the 'city' likes your decision.
        """
        self.steps += 1
        prev_total_vehicles = self.get_total_vehicles()

        # Figure out what the user wants to do with the lights
        reward = self._execute_action(action)

        # Move traffic, add new arrivals, and tick the clocks for waiting drivers
        self._simulate_traffic_flow()
        self._generate_new_traffic()
        self._update_waiting_times()

        # Life happens: sirens appear and weather shifts
        self._spawn_emergencies()
        self._maybe_change_weather()

        # Calculate the final score for this specific turn
        reward += self._calculate_step_reward(prev_total_vehicles, action)

        # Log it for the history books
        self.reward_history.append(reward)

        # The day ends if we hit the step limit or if the intersection is totally jammed
        done = self.steps >= self.max_steps or self._is_gridlocked()

        # Package up the stats
        info = self._get_info()
        
        reward = reward / 10.0
        return self.state(), reward, done, info

    def _execute_action(self, action: str) -> float:
        """Translates high-level commands into actual signal changes."""
        reward = 0.0

        if action == "RED":
            self.signal_state = SignalState.RED
            self.active_direction = None
            reward -= 2.0  # Everybody hates a 4-way red; productivity drops

        elif action == "HOLD":
            # Don't touch anything—just keep the current light green
            if self.active_direction is not None:
                reward += 0.5  # Bonus for not being 'trigger happy' with the switch
            else:
                reward -= 1.0  # You can't hold a signal that isn't on!

        elif action.startswith("GREEN_"):
            direction_str = action.replace("GREEN_", "").lower()
            try:
                direction = Direction(direction_str)
                reward += self._set_green_signal(direction)
            except ValueError:
                reward -= 5.0  # Logic error: that's not a real direction

        elif action == "PRIORITY_GREEN":
            # The 'easy button': find the messiest lane and clear it out
            priority_lane = self._get_highest_priority_lane()
            reward += self._set_green_signal(priority_lane)

        return reward

    def _set_green_signal(self, direction: Direction) -> float:
        """Flips the light to green for a specific direction and lets cars flow."""
        reward = 0.0

        # Changing lights too often causes confusion and slows things down
        if self.active_direction != direction:
            self.signal_changes += 1
            reward -= 0.5  
            if self.steps > 1 and self.signal_changes > self.steps * 0.3:
                reward -= 1.0  # Penalty for flickering the lights like a strobe

        self.active_direction = direction
        self.signal_state = SignalState.GREEN

        lane = self.lanes[direction]

        # Check how many cars can actually move (snow is slower than sunshine)
        flow_rate = self.BASE_FLOW_RATE.get(self.weather, 4)

        # If there's an emergency, other cars pull over—let's get that siren through!
        if lane.emergency:
            flow_rate = min(lane.vehicles, flow_rate + 3)
            self.emergencies_handled += 1
            reward += 15.0  # Massive gold star for letting the ambulance through
            lane.emergency = False

        # Move the vehicles out of the lane and into the 'cleared' tally
        vehicles_cleared = min(lane.vehicles, flow_rate)
        lane.vehicles -= vehicles_cleared
        self.total_vehicles_cleared += vehicles_cleared

        # Points for every car that gets to go home
        reward += vehicles_cleared * 1.5

        # Extra points for clearing out people who have been waiting a long time
        if lane.waiting_time > 5:
            reward += vehicles_cleared * 0.5

        return reward

    def _simulate_traffic_flow(self) -> None:
        """Simulate the passage of time; if you aren't moving, you're waiting."""
        for direction, lane in self.lanes.items():
            if direction != self.active_direction:
                lane.waiting_time += 1

    def _generate_new_traffic(self) -> None:
        """The 'Source' of cars: creates new arrivals based on time and weather."""
        # Check if we are in the morning or evening commute
        is_rush_hour = (7 <= self.time_of_day <= 9) or (17 <= self.time_of_day <= 19)
        multiplier = self.RUSH_HOUR_MULTIPLIER if is_rush_hour else 1.0

        # Harder modes have random 'spikes'—like a stadium letting out
        if self.task in ["medium", "hard"]:
            spike_chance = 0.15 if self.task == "hard" else 0.1
            if self.rng.random() < spike_chance:
                multiplier *= 2.0 

        for direction, lane in self.lanes.items():
            base_arrival = self.rng.randint(1, 3)
            arrivals = int(base_arrival * multiplier)

            # People tend to drive rather than walk/cycle in bad weather
            if self.weather in [Weather.RAIN, Weather.SNOW]:
                arrivals += self.rng.randint(0, 2)

            lane.vehicles += arrivals

            # The road only has so much physical space
            lane.vehicles = min(lane.vehicles, 50)

        # Tick-tock: time moves forward
        self.time_of_day = (self.time_of_day + 1) % 24

    def _update_waiting_times(self) -> None:
        """Adds to the global 'frustration' counter based on current car counts."""
        for lane in self.lanes.values():
            self.total_waiting_time += lane.vehicles * 0.1

    def _spawn_emergencies(self) -> None:
        """Occasionally, a siren will appear in a random lane."""
        emergency_chance = {
            "easy": 0.05,
            "medium": 0.1,
            "hard": 0.15
        }.get(self.task, 0.1)

        if self.rng.random() < emergency_chance:
            direction = self.rng.choice(list(Direction))
            self.lanes[direction].emergency = True

    def _maybe_change_weather(self) -> None:
        """If you're playing on 'hard', the weather can turn sour mid-game."""
        if self.task == "hard" and self.rng.random() < 0.05:
            self.weather = self.rng.choice(list(Weather))

    def _get_highest_priority_lane(self) -> Direction:
        """A 'greedy' helper that identifies which lane is currently the biggest problem."""
        best_direction = Direction.NORTH
        best_score = -float('inf')

        for direction, lane in self.lanes.items():
            score = lane.get_priority_score()
            if score > best_score:
                best_score = score
                best_direction = direction

        return best_direction

    def _calculate_step_reward(self, prev_vehicles: int, action: str) -> float:
        """The complex math that decides if the traffic controller is doing a good job."""
        reward = 0.0
        current_vehicles = self.get_total_vehicles()

        # Main Goal: Fewer cars on screen is always better
        vehicle_delta = prev_vehicles - current_vehicles
        reward += vehicle_delta * 2.0

        # Scale the penalty: a few cars is fine, a hundred is a disaster
        if current_vehicles > 80:
            reward -= 5.0
            self.congestion_events += 1
        elif current_vehicles > 60:
            reward -= 3.0
        elif current_vehicles > 40:
            reward -= 1.0

        # Don't let people sit for too long or they'll start taking it out on the city council
        avg_waiting = self._get_average_waiting_time()
        if avg_waiting > 10:
            reward -= (avg_waiting - 10) * 0.5

        # Making an ambulance wait is a huge failure
        for lane in self.lanes.values():
            if lane.emergency and lane.waiting_time > 3:
                reward -= 5.0
                self.emergencies_missed += 1

        # Efficiency: clear as many cars as possible per unit of time
        efficiency = self.total_vehicles_cleared / max(self.steps, 1)
        if efficiency > 2.0:
            reward += 1.0

        return reward

    def _is_gridlocked(self) -> bool:
        """Returns True if the traffic is so bad that the intersection is effectively dead."""
        total_vehicles = self.get_total_vehicles()
        if total_vehicles > 150:
            return True

        # If three directions are completely packed, it's game over
        gridlocked_lanes = sum(1 for lane in self.lanes.values() if lane.vehicles > 30)
        return gridlocked_lanes >= 3

    def _get_average_waiting_time(self) -> float:
        """Finds the mean wait time across all four directions."""
        total = sum(lane.waiting_time for lane in self.lanes.values())
        return total / len(self.lanes)

    def get_total_vehicles(self) -> int:
        """Simple count of every car currently visible in the intersection."""
        return sum(lane.vehicles for lane in self.lanes.values())

    def _get_info(self) -> Dict[str, Any]:
        """Summary of the current session's highs and lows."""
        return {
            "vehicles_cleared": self.total_vehicles_cleared,
            "emergencies_handled": self.emergencies_handled,
            "emergencies_missed": self.emergencies_missed,
            "congestion_events": self.congestion_events,
            "avg_waiting_time": self._get_average_waiting_time(),
            "signal_changes": self.signal_changes,
            "efficiency": self.total_vehicles_cleared / max(self.steps, 1)
        }

    def state(self) -> Dict[str, Any]:
        """Provides a snapshot of exactly what's happening for the AI (or user) to see."""
        lane_states = {}
        for direction in Direction:
            lane = self.lanes[direction]
            lane_states[direction.value] = {
                "vehicles": lane.vehicles,
                "emergency": lane.emergency,
                "waiting_time": lane.waiting_time
            }

        return {
            "lanes": lane_states,
            "active_signal": self.active_direction.value if self.active_direction else None,
            "signal_state": self.signal_state.value,
            "weather": self.weather.value,
            "time_of_day": self.time_of_day,
            "total_vehicles": self.get_total_vehicles(),
            "steps": self.steps,
            "task": self.task
        }