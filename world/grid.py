"""
Simple grid world for the self-evolving agent.
"""

import pygame
from typing import List, Tuple, Optional
from enum import Enum


class CellType(Enum):
    """Types of cells in the grid."""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3
    START = 4
    VISITED = 5


class GridWorld:
    """
    Simple grid world environment for the agent.
    """

    def __init__(self, width: int = 10, height: int = 10, cell_size: int = 50):
        """
        Initialize the grid world.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in pixels
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size

        # Initialize empty grid
        self.grid = [[CellType.EMPTY for _ in range(width)] for _ in range(height)]

        # Colors for different cell types
        self.colors = {
            CellType.EMPTY: (255, 255, 255),    # White
            CellType.WALL: (64, 64, 64),        # Dark Gray
            CellType.GOAL: (0, 255, 0),         # Green
            CellType.AGENT: (0, 100, 255),      # Blue
            CellType.START: (255, 200, 100),    # Light Orange
            CellType.VISITED: (220, 220, 220)   # Light Gray
        }

        # Create some walls for interesting navigation
        self._create_walls()

    def _create_walls(self):
        """Create walls in the grid - currently disabled for testing."""
        # No walls for now - completely open grid for testing
        pass

    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if position is valid, False otherwise
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        return self.grid[y][x] not in [CellType.WALL]

    def get_cell_type(self, x: int, y: int) -> Optional[CellType]:
        """
        Get the cell type at a position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            CellType at position, or None if out of bounds
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None

        return self.grid[y][x]

    def set_cell_type(self, x: int, y: int, cell_type: CellType):
        """
        Set the cell type at a position.

        Args:
            x: X coordinate
            y: Y coordinate
            cell_type: Type to set
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = cell_type

    def clear_agent_positions(self):
        """Clear all agent positions from the grid."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.AGENT:
                    self.grid[y][x] = CellType.EMPTY

    def clear_goal_positions(self):
        """Clear all goal positions from the grid."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.GOAL:
                    self.grid[y][x] = CellType.EMPTY

    def clear_visited_positions(self):
        """Clear all visited positions from the grid."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.VISITED:
                    self.grid[y][x] = CellType.EMPTY

    def clear_start_positions(self):
        """Clear all start positions from the grid."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.START:
                    self.grid[y][x] = CellType.EMPTY

    def clear_all_special_positions(self):
        """Clear all special positions (agent, goal, start, visited) from the grid."""
        self.clear_agent_positions()
        self.clear_goal_positions()
        self.clear_start_positions()
        self.clear_visited_positions()

    def render(self, screen: pygame.Surface):
        """
        Render the grid to a pygame surface.

        Args:
            screen: Pygame surface to render to
        """
        for y in range(self.height):
            for x in range(self.width):
                cell_type = self.grid[y][x]
                color = self.colors[cell_type]

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # Black border

    def get_grid_as_list(self) -> List[List[str]]:
        """
        Get the grid as a list of lists with string representations.

        Returns:
            2D list representing the grid
        """
        symbol_map = {
            CellType.EMPTY: ".",
            CellType.WALL: "#",
            CellType.GOAL: "G",
            CellType.AGENT: "A",
            CellType.START: "S",
            CellType.VISITED: "V"
        }

        return [[symbol_map[cell] for cell in row] for row in self.grid]

    def find_empty_position(self) -> Tuple[int, int]:
        """
        Find a random empty position in the grid.

        Returns:
            Tuple of (x, y) coordinates of an empty position
        """
        import random

        empty_positions = []
        for y in range(1, self.height - 1):  # Avoid borders
            for x in range(1, self.width - 1):
                if self.grid[y][x] == CellType.EMPTY:
                    empty_positions.append((x, y))

        if empty_positions:
            return random.choice(empty_positions)
        else:
            return (1, 1)  # Fallback position
