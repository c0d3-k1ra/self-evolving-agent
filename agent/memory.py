"""
Agent memory module for the self-evolving agent.
Tracks positions, tools, thoughts, and movement history.
"""

from typing import List, Tuple
from datetime import datetime


class AgentMemory:
    """
    Manages the agent's memory including positions, tools, and thoughts.
    """

    def __init__(self):
        """Initialize the agent memory with default values."""
        self.positions: List[dict] = []  # Changed to dict format for reflection compatibility
        self.tools: List[str] = ["move"]  # Start with basic move tool
        self.thoughts: List[str] = []
        self.moves: List[dict] = []  # Renamed from move_history for consistency
        self.reflections: List[dict] = []  # Store reflection data

    def log_position(self, x: int, y: int):
        """
        Log a new position to the agent's position history.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        position = {
            "x": x,
            "y": y,
            "timestamp": timestamp
        }
        self.positions.append(position)

    def add_tool(self, tool_name: str):
        """
        Add a new tool to the agent's toolkit.

        Args:
            tool_name: Name of the tool to add
        """
        if tool_name not in self.tools:
            self.tools.append(tool_name)

    def log_thought(self, text: str):
        """
        Log a thought or reasoning from the LLM.

        Args:
            text: The thought or reasoning text
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        thought_entry = f"[{timestamp}] {text}"
        self.thoughts.append(thought_entry)

    def log_move(self, direction: str, reason: str):
        """
        Log a movement with its direction and reasoning.

        Args:
            direction: Direction of movement (up, down, left, right, etc.)
            reason: Reason for making this move
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        move_entry = {
            "timestamp": timestamp,
            "direction": direction,
            "reason": reason
        }
        self.moves.append(move_entry)

    def log_reflection(self, reflection_dict: dict):
        """
        Log a reflection from the ReflectionManager.

        Args:
            reflection_dict: Dictionary containing reflection data
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        reflection_entry = {
            "timestamp": timestamp,
            **reflection_dict  # Include all reflection data
        }
        self.reflections.append(reflection_entry)

        # Also log as a thought for visibility
        diagnosis = reflection_dict.get("diagnosis", "No diagnosis")
        strategy = reflection_dict.get("next_strategy", "No strategy")
        self.log_thought(f"REFLECTION: {diagnosis} | Strategy: {strategy}")

    def get_summary(self) -> str:
        """
        Get a formatted summary of all logged information.

        Returns:
            Nicely formatted string containing all memory data
        """
        summary_lines = []
        summary_lines.append("=== AGENT MEMORY SUMMARY ===")
        summary_lines.append("")

        # Position history
        summary_lines.append("POSITION HISTORY:")
        if self.positions:
            for i, pos in enumerate(self.positions):
                summary_lines.append(f"  {i+1}. ({pos['x']}, {pos['y']}) [{pos['timestamp']}]")
        else:
            summary_lines.append("  No positions logged yet")
        summary_lines.append("")

        # Available tools
        summary_lines.append("AVAILABLE TOOLS:")
        for tool in self.tools:
            summary_lines.append(f"  - {tool}")
        summary_lines.append("")

        # Move history
        summary_lines.append("MOVEMENT HISTORY:")
        if self.moves:
            for move in self.moves:
                summary_lines.append(f"  [{move['timestamp']}] {move['direction']} - {move['reason']}")
        else:
            summary_lines.append("  No moves logged yet")
        summary_lines.append("")

        # Reflections
        summary_lines.append("REFLECTIONS:")
        if self.reflections:
            for reflection in self.reflections:
                summary_lines.append(f"  [{reflection['timestamp']}] {reflection.get('diagnosis', 'No diagnosis')}")
                summary_lines.append(f"    Strategy: {reflection.get('next_strategy', 'No strategy')}")
                summary_lines.append(f"    Corrected Move: {reflection.get('corrected_move', 'None')}")
        else:
            summary_lines.append("  No reflections yet")
        summary_lines.append("")

        # Thoughts
        summary_lines.append("THOUGHTS & REASONING:")
        if self.thoughts:
            for thought in self.thoughts:
                summary_lines.append(f"  {thought}")
        else:
            summary_lines.append("  No thoughts logged yet")
        summary_lines.append("")

        summary_lines.append("=== END SUMMARY ===")
        return "\n".join(summary_lines)

    def get_recent_positions(self, count: int = 5) -> List[Tuple[int, int]]:
        """
        Get the most recent positions.

        Args:
            count: Number of recent positions to return

        Returns:
            List of recent positions
        """
        return self.positions[-count:] if self.positions else []

    def get_position_count(self) -> int:
        """
        Get the total number of positions logged.

        Returns:
            Number of positions in history
        """
        return len(self.positions)

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is available.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is available, False otherwise
        """
        return tool_name in self.tools

    def clear_memory(self):
        """Clear all memory data (useful for resets)."""
        self.positions.clear()
        self.tools = ["move"]  # Reset to default tools
        self.thoughts.clear()
        self.moves.clear()
        self.reflections.clear()

    def get_stats(self) -> dict:
        """
        Get basic statistics about the agent's memory.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "total_positions": len(self.positions),
            "total_tools": len(self.tools),
            "total_thoughts": len(self.thoughts),
            "total_moves": len(self.moves),
            "total_reflections": len(self.reflections)
        }

    def get_last_reflection(self) -> dict:
        """
        Get the most recent reflection.

        Returns:
            Last reflection dictionary, or empty dict if none exist
        """
        return self.reflections[-1] if self.reflections else {}
