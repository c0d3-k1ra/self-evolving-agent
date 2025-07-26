"""
Agent implementation for the self-evolving agent.
Uses LLM-based decision making and integrates with AgentMemory.
"""

import os
import json
from typing import Tuple, Dict, Any, Optional
from openai import OpenAI
from .memory import AgentMemory


class Agent:
    """
    Intelligent agent that uses LLM for decision making and tracks memory.
    """

    def __init__(self, goal: Tuple[int, int], memory: AgentMemory, current_position: Tuple[int, int] = (0, 0)):
        """
        Initialize the agent with goal and memory.

        Args:
            goal: Target position (gx, gy)
            memory: AgentMemory instance for logging
            current_position: Starting position (x, y)
        """
        self.goal = goal
        self.memory = memory
        self.current_position = current_position

        # Path planning state
        self.planned_path = []  # List of moves to execute
        self.path_index = 0     # Current position in planned path
        self.position_history = []  # Track recent positions for loop detection
        self.stuck_counter = 0  # Count consecutive failed attempts
        self.max_stuck_attempts = 3  # Re-plan after this many stuck attempts
        self.position_history_size = 10  # Keep track of last N positions

        # Initialize OpenAI client with custom URL and model support
        self.client = None
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')  # Default to gpt-3.5-turbo

        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')  # Custom URL for API endpoint

        if api_key:
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"Using custom OpenAI endpoint: {base_url} with model: {self.model}")
            else:
                self.client = OpenAI(api_key=api_key)
                print(f"Using OpenAI API with model: {self.model}")
        else:
            print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            print("Optional: Set OPENAI_BASE_URL for custom endpoints and OPENAI_MODEL for custom models.")

        # Log initial position
        self.memory.log_position(current_position[0], current_position[1])
        self.position_history.append(current_position)

    def direction_to_delta(self, direction: str) -> Tuple[int, int]:
        """
        Convert direction string to coordinate delta.

        Args:
            direction: Direction string (up, down, left, right)

        Returns:
            Tuple of (dx, dy) coordinate changes
        """
        direction_map = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        return direction_map.get(direction.lower(), (0, 0))

    def _format_grid_context(self, grid: Any) -> str:
        """
        Format grid information for LLM context.

        Args:
            grid: Grid representation (could be 2D array, string, etc.)

        Returns:
            String representation of grid context
        """
        # Handle different grid formats
        if isinstance(grid, list):
            # 2D array format
            grid_str = "Grid layout:\n"
            for row in grid:
                grid_str += " ".join(str(cell) for cell in row) + "\n"
            return grid_str
        elif isinstance(grid, str):
            # String format
            return f"Grid context: {grid}"
        else:
            # Generic format
            return f"Grid information: {str(grid)}"

    def _create_path_planning_prompt(self, grid: Any) -> str:
        """
        Create the prompt for LLM path planning.

        Args:
            grid: Current grid state

        Returns:
            Formatted prompt string for path planning
        """
        x, y = self.current_position
        gx, gy = self.goal

        # Get recent movement history for context
        recent_positions = self.position_history[-5:] if self.position_history else []
        position_context = f"Recent positions: {recent_positions}" if recent_positions else "No recent movement history"

        # Check if we've been stuck
        stuck_context = ""
        if self.stuck_counter > 0:
            stuck_context = f"IMPORTANT: Previous attempts failed {self.stuck_counter} times. You need a different strategy!"

        grid_context = self._format_grid_context(grid)

        prompt = f"""You are an intelligent agent navigating a grid world. You need to plan a complete path from your current position to the goal.

Current situation:
- Your position: ({x}, {y})
- Goal position: ({gx}, {gy})
- {position_context}
{stuck_context}

{grid_context}

COORDINATE SYSTEM:
- X increases from LEFT to RIGHT (0 = leftmost, higher = rightmost)
- Y increases from TOP to BOTTOM (0 = topmost, higher = bottommost)
- "up" = decrease Y (move towards top)
- "down" = increase Y (move towards bottom)
- "left" = decrease X (move towards left)
- "right" = increase X (move towards right)

Legend: '.' = empty space, '#' = wall/obstacle, 'G' = goal, 'A' = your current position

TASK: Plan a complete sequence of moves to reach the goal efficiently.

Available moves: up, down, left, right

MOVEMENT RULES:
- To reach a HIGHER Y coordinate (goal Y > current Y): use "down"
- To reach a LOWER Y coordinate (goal Y < current Y): use "up"
- To reach a HIGHER X coordinate (goal X > current X): use "right"
- To reach a LOWER X coordinate (goal X < current X): use "left"

Please respond with a JSON object containing:
- "path": array of moves like ["right", "right", "down", "left", "up"]
- "reasoning": explanation of your path planning strategy
- "estimated_steps": number of moves in your path

Example response:
{{"path": ["right", "down", "right"], "reasoning": "Moving around obstacle to reach goal", "estimated_steps": 3}}

IMPORTANT:
- Plan the COMPLETE path, not just the next move
- Consider the coordinate system: Y=0 is TOP, Y increases DOWNWARD
- If goal Y > current Y, you need to move DOWN
- If goal Y < current Y, you need to move UP
"""
        return prompt

    def _detect_loop(self) -> bool:
        """
        Detect if the agent is stuck in a loop by checking recent positions.

        Returns:
            True if stuck in a loop, False otherwise
        """
        if len(self.position_history) < 4:
            return False

        # Count how many times current position appears in recent history
        current_pos = self.current_position
        recent_positions = self.position_history[-6:]  # Check last 6 positions
        position_count = recent_positions.count(current_pos)

        # If we've been in the same position 3+ times recently, we're stuck
        return position_count >= 3

    def _plan_path(self, grid: Any) -> bool:
        """
        Use LLM to plan a complete path from current position to goal.

        Args:
            grid: Current grid state

        Returns:
            True if path planning succeeded, False otherwise
        """
        if not self.client:
            # Use fallback path planning
            return self._fallback_path_planning()

        try:
            prompt = self._create_path_planning_prompt(grid)

            self.memory.log_thought(f"Planning path from {self.current_position} to {self.goal}")

            # Print LLM call details
            print("\n" + "="*80)
            print("🤖 LLM PATH PLANNING CALL")
            print("="*80)
            print(f"Model: {self.model}")
            print(f"Temperature: 0.3")
            print(f"Max Tokens: 300")
            print("\nPROMPT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that plans navigation paths. Always respond with valid JSON containing a complete path."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=300
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()

            # Print LLM response
            print("\nLLM RESPONSE:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            print("="*80)

            try:
                # Extract JSON from response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    path_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                self.memory.log_thought(f"Failed to parse LLM path planning response: {response_text}")
                return self._fallback_path_planning()

            # Validate the path
            path = path_data.get("path", [])
            reasoning = path_data.get("reasoning", "No reasoning provided")

            if not isinstance(path, list) or len(path) == 0:
                self.memory.log_thought("Invalid path from LLM, using fallback")
                return self._fallback_path_planning()

            # Validate each move in the path
            valid_moves = ["up", "down", "left", "right"]
            for move in path:
                if move.lower() not in valid_moves:
                    self.memory.log_thought(f"Invalid move '{move}' in path, using fallback")
                    return self._fallback_path_planning()

            # Set the planned path
            self.planned_path = [move.lower() for move in path]
            self.path_index = 0

            # Log the plan
            self.memory.log_thought(f"LLM planned path: {self.planned_path}")
            self.memory.log_thought(f"Path reasoning: {reasoning}")

            return True

        except Exception as e:
            self.memory.log_thought(f"Error in LLM path planning: {str(e)}")
            return self._fallback_path_planning()

    def _fallback_path_planning(self) -> bool:
        """
        Simple fallback path planning when LLM is unavailable.

        Returns:
            True (always succeeds with simple path)
        """
        x, y = self.current_position
        gx, gy = self.goal

        path = []

        # Simple path: move horizontally first, then vertically
        while x != gx:
            if gx > x:
                path.append("right")
                x += 1
            else:
                path.append("left")
                x -= 1

        while y != gy:
            if gy > y:
                path.append("down")
                y += 1
            else:
                path.append("up")
                y -= 1

        self.planned_path = path
        self.path_index = 0

        self.memory.log_thought(f"Fallback planned path: {self.planned_path}")
        return True

    def decide_next_move(self, grid: Any) -> Optional[Tuple[int, int]]:
        """
        Decide the next move using path planning approach.

        Args:
            grid: Current grid state

        Returns:
            Tuple of (dx, dy) for the next move, or None if decision fails
        """
        # Check if we need to plan a new path
        need_new_plan = (
            len(self.planned_path) == 0 or  # No current plan
            self.path_index >= len(self.planned_path) or  # Finished current plan
            self._detect_loop()  # Stuck in a loop
        )

        if need_new_plan:
            if self._detect_loop():
                self.stuck_counter += 1
                self.memory.log_thought(f"Detected loop! Stuck counter: {self.stuck_counter}")

                if self.stuck_counter >= self.max_stuck_attempts:
                    self.memory.log_thought("Too many failed attempts, resetting stuck counter")
                    self.stuck_counter = 0

            # Plan a new path
            if not self._plan_path(grid):
                return self._fallback_decision()

        # Execute the next move from the planned path
        if self.path_index < len(self.planned_path):
            next_move = self.planned_path[self.path_index]
            self.path_index += 1

            # Log the move
            self.memory.log_move(next_move, f"Following planned path (step {self.path_index}/{len(self.planned_path)})")

            return self.direction_to_delta(next_move)
        else:
            # Shouldn't happen, but fallback just in case
            return self._fallback_decision()

    def _fallback_decision(self) -> Tuple[int, int]:
        """
        Simple fallback decision when LLM is unavailable.

        Returns:
            Tuple of (dx, dy) for a basic move towards goal
        """
        x, y = self.current_position
        gx, gy = self.goal

        # Simple greedy approach: move towards goal
        dx = 0
        dy = 0

        if gx > x:
            dx = 1
            direction = "right"
        elif gx < x:
            dx = -1
            direction = "left"
        elif gy > y:
            dy = 1
            direction = "down"
        elif gy < y:
            dy = -1
            direction = "up"
        else:
            # Already at goal
            direction = "stay"

        reason = f"Fallback: Moving {direction} towards goal"
        self.memory.log_move(direction, reason)
        self.memory.log_thought(reason)

        return (dx, dy)

    def update_position(self, new_position: Tuple[int, int]):
        """
        Update the agent's current position and log it.

        Args:
            new_position: New (x, y) position
        """
        self.current_position = new_position
        self.memory.log_position(new_position[0], new_position[1])

        # Update position history for loop detection
        self.position_history.append(new_position)

        # Keep only recent positions to avoid memory bloat
        if len(self.position_history) > self.position_history_size:
            self.position_history = self.position_history[-self.position_history_size:]

        # Reset stuck counter if we successfully moved to a new position
        if len(self.position_history) >= 2 and self.position_history[-1] != self.position_history[-2]:
            if self.stuck_counter > 0:
                self.memory.log_thought("Successfully moved to new position, resetting stuck counter")
                self.stuck_counter = 0

    def get_position(self) -> Tuple[int, int]:
        """
        Get the current position.

        Returns:
            Current (x, y) position
        """
        return self.current_position

    def get_goal(self) -> Tuple[int, int]:
        """
        Get the goal position.

        Returns:
            Goal (x, y) position
        """
        return self.goal

    def set_goal(self, new_goal: Tuple[int, int]):
        """
        Set a new goal position.

        Args:
            new_goal: New goal (x, y) position
        """
        self.goal = new_goal
        self.memory.log_thought(f"Goal changed to {new_goal}")

    def is_at_goal(self) -> bool:
        """
        Check if the agent has reached the goal.

        Returns:
            True if at goal position, False otherwise
        """
        return self.current_position == self.goal

    def get_distance_to_goal(self) -> int:
        """
        Calculate Manhattan distance to goal.

        Returns:
            Manhattan distance to goal
        """
        x, y = self.current_position
        gx, gy = self.goal
        return abs(gx - x) + abs(gy - y)
