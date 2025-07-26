"""
Reflection system for the self-evolving agent.
Enables the agent to detect when it's stuck and reflect on better strategies.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
from .memory import AgentMemory


class ReflectionManager:
    """
    Manages reflection and meta-cognitive analysis for the agent.
    """

    def __init__(self):
        """Initialize the reflection manager with OpenAI client."""
        # Initialize OpenAI client (same setup as Agent)
        self.client = None
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')

        if api_key:
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        else:
            print("Warning: ReflectionManager - No OpenAI API key found.")

    def should_reflect(self, memory: AgentMemory, goal: Tuple[int, int], current_position: Tuple[int, int]) -> bool:
        """
        Determine if the agent should reflect on its performance.

        Args:
            memory: Agent's memory system
            goal: Goal position
            current_position: Current agent position

        Returns:
            True if reflection is needed, False otherwise
        """
        # Get recent positions from memory
        positions = memory.positions

        if len(positions) < 5:
            return False  # Not enough data to reflect

        # Criterion 1: More than 15 steps without reaching goal
        step_count = len(positions)
        if step_count > 15 and current_position != goal:
            print(f"🤔 Reflection trigger: {step_count} steps taken without reaching goal")
            return True

        # Criterion 2: Same position occurred 3+ times in last 10 moves
        recent_positions = positions[-10:] if len(positions) >= 10 else positions
        position_counts = {}

        for pos in recent_positions:
            pos_tuple = (pos['x'], pos['y'])
            position_counts[pos_tuple] = position_counts.get(pos_tuple, 0) + 1

        max_repeats = max(position_counts.values()) if position_counts else 0
        if max_repeats >= 3:
            print(f"🤔 Reflection trigger: Position repeated {max_repeats} times in recent moves")
            return True

        # Criterion 3 (Optional): Average distance hasn't decreased in 5 moves
        if len(positions) >= 5:
            recent_5_positions = positions[-5:]
            distances = []

            for pos in recent_5_positions:
                distance = abs(goal[0] - pos['x']) + abs(goal[1] - pos['y'])
                distances.append(distance)

            # Check if distance is stagnating (not decreasing on average)
            if len(distances) >= 5:
                first_half_avg = sum(distances[:2]) / 2
                second_half_avg = sum(distances[-2:]) / 2

                if second_half_avg >= first_half_avg:
                    print(f"🤔 Reflection trigger: Distance not improving (was {first_half_avg:.1f}, now {second_half_avg:.1f})")
                    return True

        return False

    def generate_reflection(self, memory: AgentMemory, goal: Tuple[int, int], current_position: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Generate a reflection using LLM analysis.

        Args:
            memory: Agent's memory system
            goal: Goal position
            current_position: Current agent position

        Returns:
            Dictionary with reflection data, or None if generation fails
        """
        if not self.client:
            return self._fallback_reflection(memory, goal, current_position)

        try:
            # Prepare reflection context
            x, y = current_position
            gx, gy = goal

            # Get recent positions (last 10)
            recent_positions = memory.positions[-10:] if len(memory.positions) >= 10 else memory.positions
            position_list = [f"({pos['x']}, {pos['y']})" for pos in recent_positions]

            # Get recent thoughts (last 5)
            recent_thoughts = memory.thoughts[-5:] if len(memory.thoughts) >= 5 else memory.thoughts

            # Get recent moves (last 5)
            recent_moves = memory.moves[-5:] if len(memory.moves) >= 5 else memory.moves
            move_descriptions = [f"{move['direction']} - {move['reason']}" for move in recent_moves]

            step_count = len(memory.positions)

            # Create reflection prompt
            prompt = f"""You are an agent navigating a grid world. You are stuck or looping.

CURRENT SITUATION:
- Current position: ({x}, {y})
- Goal: ({gx}, {gy})
- Steps taken: {step_count}
- Distance to goal: {abs(gx - x) + abs(gy - y)}

RECENT POSITIONS: {', '.join(position_list)}

RECENT MOVES:
{chr(10).join(f"- {move}" for move in move_descriptions)}

RECENT THOUGHTS:
{chr(10).join(f'- "{thought}"' for thought in recent_thoughts)}

ANALYSIS TASK:
You appear to be stuck or inefficient. Reflect on your navigation strategy.

Reflect on:
1. Why you're stuck or inefficient
2. What patterns you notice in your movement
3. A better strategy or specific next move
4. How to avoid this situation in the future

Return JSON:
{{
  "diagnosis": "Detailed analysis of what's going wrong",
  "pattern_detected": "What repetitive behavior you notice",
  "next_strategy": "High-level strategy to improve",
  "corrected_move": "up/down/left/right - specific next move recommendation",
  "reason": "Why this corrected move is better",
  "future_advice": "How to avoid getting stuck again"
}}

IMPORTANT: The corrected_move must be one of: up, down, left, right"""

            print("\n" + "="*80)
            print("🧠 LLM REFLECTION CALL")
            print("="*80)
            print(f"Model: {self.model}")
            print(f"Temperature: 0.8")
            print(f"Max Tokens: 400")
            print("\nREFLECTION PROMPT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes navigation patterns and provides strategic insights. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for more creative reflection
                max_tokens=400
            )

            response_text = response.choices[0].message.content.strip()

            print("\nLLM REFLECTION RESPONSE:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            print("="*80)

            # Parse JSON response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    reflection_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse reflection JSON: {e}")
                return self._fallback_reflection(memory, goal, current_position)

            # Validate reflection data
            required_fields = ["diagnosis", "next_strategy", "corrected_move", "reason"]
            for field in required_fields:
                if field not in reflection_data:
                    print(f"Missing required field in reflection: {field}")
                    return self._fallback_reflection(memory, goal, current_position)

            # Validate corrected_move
            valid_moves = ["up", "down", "left", "right"]
            if reflection_data["corrected_move"].lower() not in valid_moves:
                print(f"Invalid corrected_move: {reflection_data['corrected_move']}")
                reflection_data["corrected_move"] = self._get_fallback_move(current_position, goal)

            # Add metadata
            reflection_data["timestamp"] = len(memory.positions)
            reflection_data["trigger_position"] = current_position
            reflection_data["goal_at_reflection"] = goal

            return reflection_data

        except Exception as e:
            print(f"Error generating reflection: {e}")
            return self._fallback_reflection(memory, goal, current_position)

    def _fallback_reflection(self, memory: AgentMemory, goal: Tuple[int, int], current_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Generate a simple fallback reflection when LLM is unavailable.

        Args:
            memory: Agent's memory system
            goal: Goal position
            current_position: Current agent position

        Returns:
            Basic reflection dictionary
        """
        x, y = current_position
        gx, gy = goal

        # Simple analysis
        distance = abs(gx - x) + abs(gy - y)
        step_count = len(memory.positions)

        # Determine best direction
        corrected_move = self._get_fallback_move(current_position, goal)

        return {
            "diagnosis": f"Fallback analysis: Taken {step_count} steps, still {distance} units from goal",
            "pattern_detected": "Unable to analyze patterns without LLM",
            "next_strategy": "Use simple greedy approach toward goal",
            "corrected_move": corrected_move,
            "reason": f"Fallback: Move {corrected_move} to get closer to goal",
            "future_advice": "Try to move more directly toward the goal",
            "timestamp": step_count,
            "trigger_position": current_position,
            "goal_at_reflection": goal
        }

    def _get_fallback_move(self, current_position: Tuple[int, int], goal: Tuple[int, int]) -> str:
        """
        Get a simple fallback move toward the goal.

        Args:
            current_position: Current position
            goal: Goal position

        Returns:
            Direction string (up/down/left/right)
        """
        x, y = current_position
        gx, gy = goal

        # Simple greedy choice
        if abs(gx - x) > abs(gy - y):
            # Prioritize horizontal movement
            return "right" if gx > x else "left"
        else:
            # Prioritize vertical movement
            return "down" if gy > y else "up"
