"""
Main execution script for the self-evolving agent.
Runs the pygame loop with LLM-based agent navigation.
"""

import pygame
import sys
import os
import time
from dotenv import load_dotenv
from agent.memory import AgentMemory
from agent.agent import Agent
from world.grid import GridWorld, CellType

# Load environment variables from .env file
load_dotenv()


class AgentSimulation:
    """
    Main simulation class that runs the agent in the grid world.
    """

    def __init__(self):
        """Initialize the simulation."""
        # Initialize pygame
        pygame.init()

        # Create grid world
        self.grid_world = GridWorld(width=12, height=10, cell_size=60)

        # Create pygame screen
        self.screen = pygame.display.set_mode(
            (self.grid_world.screen_width, self.grid_world.screen_height + 100)  # Extra space for info
        )
        pygame.display.set_caption("Self-Evolving Agent")

        # Create font for text display
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Initialize agent
        self.memory = AgentMemory()

        # Find starting position and goal
        start_pos = self.grid_world.find_empty_position()
        goal_pos = self.grid_world.find_empty_position()

        # Make sure goal is different from start
        while goal_pos == start_pos:
            goal_pos = self.grid_world.find_empty_position()

        self.agent = Agent(goal=goal_pos, memory=self.memory, current_position=start_pos)

        # Set goal in grid
        self.grid_world.set_cell_type(goal_pos[0], goal_pos[1], CellType.GOAL)

        # Simulation state
        self.running = True
        self.paused = False
        self.step_delay = 1.0  # Seconds between moves
        self.last_move_time = 0
        self.move_count = 0
        self.max_moves = 100

        print(f"Agent starting at {start_pos}, goal at {goal_pos}")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset simulation")
        print("  Q - Quit")
        print("  UP/DOWN - Adjust speed")

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

                elif event.key == pygame.K_r:
                    self.reset_simulation()

                elif event.key == pygame.K_q:
                    self.running = False

                elif event.key == pygame.K_UP:
                    self.step_delay = max(0.1, self.step_delay - 0.2)
                    print(f"Speed increased (delay: {self.step_delay:.1f}s)")

                elif event.key == pygame.K_DOWN:
                    self.step_delay = min(3.0, self.step_delay + 0.2)
                    print(f"Speed decreased (delay: {self.step_delay:.1f}s)")

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        print("Resetting simulation...")

        # Clear ALL positions from grid (agents and goals)
        self.grid_world.clear_agent_positions()
        self.grid_world.clear_goal_positions()  # Clear old goals too!

        # Find new positions
        start_pos = self.grid_world.find_empty_position()
        goal_pos = self.grid_world.find_empty_position()

        while goal_pos == start_pos:
            goal_pos = self.grid_world.find_empty_position()

        # Reset agent
        self.memory.clear_memory()
        self.agent = Agent(goal=goal_pos, memory=self.memory, current_position=start_pos)

        # Set new goal
        self.grid_world.set_cell_type(goal_pos[0], goal_pos[1], CellType.GOAL)

        # Reset counters
        self.move_count = 0
        self.last_move_time = 0

        print(f"New start: {start_pos}, new goal: {goal_pos}")

    def update_agent(self):
        """Update agent position based on LLM decision."""
        if self.paused or self.agent.is_at_goal():
            return

        current_time = time.time()
        if current_time - self.last_move_time < self.step_delay:
            return

        if self.move_count >= self.max_moves:
            print("Maximum moves reached! Resetting...")
            self.reset_simulation()
            return

        # Clear previous agent position from grid
        self.grid_world.clear_agent_positions()

        # Get agent decision
        grid_data = self.grid_world.get_grid_as_list()
        decision = self.agent.decide_next_move(grid_data)

        if decision:
            dx, dy = decision
            current_x, current_y = self.agent.get_position()
            new_x, new_y = current_x + dx, current_y + dy

            # Check if move is valid
            if self.grid_world.is_valid_position(new_x, new_y):
                # Don't move into goal cell (we'll handle goal reaching separately)
                if self.grid_world.get_cell_type(new_x, new_y) != CellType.GOAL:
                    self.agent.update_position((new_x, new_y))
                else:
                    # Reached goal!
                    self.agent.update_position((new_x, new_y))
                    print(f"🎉 Goal reached in {self.move_count + 1} moves!")
                    print("Agent memory summary:")
                    print(self.memory.get_summary())
                    time.sleep(2)  # Pause to show success
                    self.reset_simulation()
                    return
            else:
                print(f"Invalid move attempted: ({current_x}, {current_y}) -> ({new_x}, {new_y})")

        # Update grid with agent position
        agent_pos = self.agent.get_position()
        self.grid_world.set_cell_type(agent_pos[0], agent_pos[1], CellType.AGENT)

        self.move_count += 1
        self.last_move_time = current_time

    def render_info(self):
        """Render information text below the grid."""
        y_offset = self.grid_world.screen_height + 10

        # Agent info
        pos = self.agent.get_position()
        goal = self.agent.get_goal()
        distance = self.agent.get_distance_to_goal()

        info_lines = [
            f"Position: {pos}  Goal: {goal}  Distance: {distance}",
            f"Moves: {self.move_count}/{self.max_moves}  Speed: {self.step_delay:.1f}s",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}  {'GOAL REACHED!' if self.agent.is_at_goal() else ''}"
        ]

        for i, line in enumerate(info_lines):
            text_surface = self.small_font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset + i * 20))

        # Recent thoughts
        recent_thoughts = self.memory.thoughts[-2:] if self.memory.thoughts else []
        if recent_thoughts:
            thought_y = y_offset + 70
            thought_text = self.small_font.render("Recent thoughts:", True, (0, 0, 100))
            self.screen.blit(thought_text, (10, thought_y))

            for i, thought in enumerate(recent_thoughts):
                # Truncate long thoughts
                display_thought = thought[:80] + "..." if len(thought) > 80 else thought
                thought_surface = self.small_font.render(display_thought, True, (0, 0, 150))
                self.screen.blit(thought_surface, (10, thought_y + 20 + i * 15))

    def render(self):
        """Render the entire simulation."""
        # Clear screen
        self.screen.fill((240, 240, 240))

        # Render grid
        self.grid_world.render(self.screen)

        # Render info
        self.render_info()

        # Update display
        pygame.display.flip()

    def run(self):
        """Main simulation loop."""
        clock = pygame.time.Clock()

        while self.running:
            self.handle_events()
            self.update_agent()
            self.render()
            clock.tick(60)  # 60 FPS

        pygame.quit()
        sys.exit()


def main():
    """Main entry point."""
    print("Starting Self-Evolving Agent Simulation...")
    print("Make sure to set your OpenAI API key:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("Optional environment variables:")
    print("  export OPENAI_MODEL='gpt-4'  # Default: gpt-3.5-turbo")
    print("  export OPENAI_BASE_URL='http://localhost:8000/v1'  # For custom endpoints")
    print()

    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Agent will use fallback navigation.")
        print("   Set the API key to enable LLM-based decision making.")
        print()

    try:
        simulation = AgentSimulation()
        simulation.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
