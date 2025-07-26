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

    def decide_next_move_stepwise(self, grid, goal=None, memory=None) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Step-by-step LLM decision making for next move.

        Args:
            grid: Current grid state (2D list or string representation)
            goal: Goal position (optional, uses self.goal if not provided)
            memory: Memory context (optional, uses self.memory if not provided)

        Returns:
            Tuple of ((dx, dy), reason) or (None, reason) if decision fails
        """
        # Use provided parameters or defaults
        goal_pos = goal if goal is not None else self.goal
        memory_ctx = memory if memory is not None else self.memory

        x, y = self.current_position
        goal_x, goal_y = goal_pos

        # Format grid as text block and mark current agent position
        if isinstance(grid, list):
            # Create a copy of the grid to modify
            grid_copy = [row[:] for row in grid]
            # Mark current agent position as 'A'
            if 0 <= y < len(grid_copy) and 0 <= x < len(grid_copy[0]):
                grid_copy[y][x] = 'A'
            grid_text = "\n".join(" ".join(str(cell) for cell in row) for row in grid_copy)

            # Get grid dimensions
            grid_height = len(grid)
            grid_width = len(grid[0]) if grid else 0
        else:
            grid_text = str(grid)
            grid_height = 10  # Default assumption
            grid_width = 12   # Default assumption

        # Determine valid moves based on boundaries
        valid_moves = []
        if x > 0:
            valid_moves.append("left")
        if x < grid_width - 1:
            valid_moves.append("right")
        if y > 0:
            valid_moves.append("up")
        if y < grid_height - 1:
            valid_moves.append("down")

        # Create boundary context
        boundary_info = []
        if x == 0:
            boundary_info.append("LEFT edge (cannot move left)")
        if x == grid_width - 1:
            boundary_info.append("RIGHT edge (cannot move right)")
        if y == 0:
            boundary_info.append("TOP edge (cannot move up)")
        if y == grid_height - 1:
            boundary_info.append("BOTTOM edge (cannot move down)")

        boundary_context = f"Boundaries: {', '.join(boundary_info)}" if boundary_info else "No boundary constraints"

        # Check for previous reflections
        last_reflection = memory_ctx.get_last_reflection()
        reflection_context = ""
        if last_reflection:
            diagnosis = last_reflection.get('diagnosis', 'No diagnosis')
            strategy = last_reflection.get('next_strategy', 'No strategy')
            advice = last_reflection.get('future_advice', 'No advice')
            reflection_context = f"""
PREVIOUS REFLECTION:
- Diagnosis: {diagnosis}
- Suggested Strategy: {strategy}
- Advice: {advice}
- Consider this reflection when making your decision.
"""

        # Create the step-by-step prompt
        prompt = f"""You are an agent navigating a grid world.

CURRENT SITUATION:
- Your position: ({x}, {y})
- Goal position: ({goal_x}, {goal_y})
- Grid size: {grid_width} columns (X: 0-{grid_width-1}), {grid_height} rows (Y: 0-{grid_height-1})
- {boundary_context}

VALID MOVES: {', '.join(valid_moves)}

Grid layout:
{grid_text}

Legend:
. = empty space, # = wall, G = goal, A = agent (YOU), S = start, V = visited

COORDINATE SYSTEM:
- X increases LEFT to RIGHT (0 = leftmost)
- Y increases TOP to BOTTOM (0 = topmost)
- up = decrease Y, down = increase Y, left = decrease X, right = increase X

{reflection_context}

TASK: Choose your next move to get closer to the goal.

IMPORTANT: You can ONLY choose from these valid moves: {', '.join(valid_moves)}

Return a JSON:
{{
  "move": "right",
  "reason": "Moving right brings me closer to the goal"
}}"""

        if not self.client:
            # Fallback decision
            return self._fallback_decision_stepwise(goal_pos)

        try:
            # Print LLM call details
            print("\n" + "="*80)
            print("🤖 LLM STEPWISE DECISION CALL")
            print("="*80)
            print(f"Model: {self.model}")
            print(f"Temperature: 0.7")
            print(f"Max Tokens: 150")
            print("\nPROMPT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that makes navigation decisions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()

            # Print LLM response
            print("\nLLM RESPONSE:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            print("="*80)

            # Extract JSON from response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    decision_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                memory_ctx.log_thought(f"Failed to parse LLM response: {response_text}")
                return self._fallback_decision_stepwise(goal_pos)

            # Validate the decision
            move = decision_data.get("move", "").lower()
            reason = decision_data.get("reason", "No reason provided")

            # Check if move is valid (both a real direction and within boundaries)
            if move not in ["up", "down", "left", "right"]:
                memory_ctx.log_thought(f"Invalid move '{move}' from LLM, using fallback")
                return self._fallback_decision_stepwise(goal_pos)

            # Check if move is within valid moves for current position
            if move not in [m.lower() for m in valid_moves]:
                memory_ctx.log_thought(f"LLM chose '{move}' but only {valid_moves} are valid from position ({x}, {y}), using fallback")
                return self._fallback_decision_stepwise(goal_pos)

            # Log the decision
            memory_ctx.log_move(move, reason)
            memory_ctx.log_thought(f"LLM decided: {move} - {reason}")

            # Convert to delta and return
            dx, dy = self.direction_to_delta(move)
            return ((dx, dy), reason)

        except Exception as e:
            memory_ctx.log_thought(f"Error in LLM stepwise decision making: {str(e)}")
            return self._fallback_decision_stepwise(goal_pos)

    def _fallback_decision_stepwise(self, goal_pos: Tuple[int, int]) -> Tuple[Tuple[int, int], str]:
        """
        Simple fallback decision for stepwise approach.

        Args:
            goal_pos: Goal position

        Returns:
            Tuple of ((dx, dy), reason)
        """
        x, y = self.current_position
        gx, gy = goal_pos

        # Simple greedy approach: move towards goal
        dx, dy = 0, 0
        direction = "stay"

        if gx > x:
            dx, dy = 1, 0
            direction = "right"
        elif gx < x:
            dx, dy = -1, 0
            direction = "left"
        elif gy > y:
            dx, dy = 0, 1
            direction = "down"
        elif gy < y:
            dx, dy = 0, -1
            direction = "up"

        reason = f"Fallback: Moving {direction} towards goal"
        self.memory.log_move(direction, reason)
        self.memory.log_thought(reason)

        return ((dx, dy), reason)

    def decide_next_move(self, grid: Any) -> Optional[Tuple[int, int]]:
        """
        Decide the next move using stepwise LLM approach.

        Args:
            grid: Current grid state

        Returns:
            Tuple of (dx, dy) for the next move, or None if decision fails
        """
        # Use the new stepwise decision method
        result, reason = self.decide_next_move_stepwise(grid)
        return result

    def update_position(self, new_position: Tuple[int, int]):
        """
        Update the agent's current position and log it.

        Args:
            new_position: New (x, y) position
        """
        self.current_position = new_position
        self.memory.log_position(new_position[0], new_position[1])

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
