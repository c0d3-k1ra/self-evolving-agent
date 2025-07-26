#!/usr/bin/env python3
"""
Simple test script to verify the agent components work correctly.
"""

import os
import sys
from dotenv import load_dotenv
from agent.memory import AgentMemory
from agent.agent import Agent
from world.grid import GridWorld, CellType

# Load environment variables from .env file
load_dotenv()


def test_memory():
    """Test the AgentMemory class."""
    print("Testing AgentMemory...")

    memory = AgentMemory()

    # Test logging
    memory.log_position(0, 0)
    memory.log_position(1, 0)
    memory.add_tool("jump")
    memory.log_thought("I should move towards the goal")
    memory.log_move("right", "Moving closer to goal")

    # Test retrieval
    assert len(memory.positions) == 2
    assert memory.has_tool("move")
    assert memory.has_tool("jump")
    assert not memory.has_tool("fly")

    # Test summary
    summary = memory.get_summary()
    assert "POSITION HISTORY" in summary
    assert "AVAILABLE TOOLS" in summary

    print("✅ AgentMemory tests passed!")
    return True


def test_grid_world():
    """Test the GridWorld class."""
    print("Testing GridWorld...")

    grid = GridWorld(width=5, height=5, cell_size=50)

    # Test basic functionality
    assert grid.width == 5
    assert grid.height == 5

    # Test position validation
    assert not grid.is_valid_position(-1, 0)  # Out of bounds
    assert not grid.is_valid_position(0, 0)   # Wall (border)
    assert grid.is_valid_position(1, 1)       # Valid empty space

    # Test cell operations
    grid.set_cell_type(2, 2, CellType.GOAL)
    assert grid.get_cell_type(2, 2) == CellType.GOAL

    # Test grid representation
    grid_list = grid.get_grid_as_list()
    assert len(grid_list) == 5
    assert len(grid_list[0]) == 5

    print("✅ GridWorld tests passed!")
    return True


def test_agent_basic():
    """Test basic Agent functionality without LLM."""
    print("Testing Agent (basic functionality)...")

    memory = AgentMemory()
    agent = Agent(goal=(3, 3), memory=memory, current_position=(1, 1))

    # Test basic properties
    assert agent.get_position() == (1, 1)
    assert agent.get_goal() == (3, 3)
    assert agent.get_distance_to_goal() == 4  # Manhattan distance
    assert not agent.is_at_goal()

    # Test direction conversion
    assert agent.direction_to_delta("up") == (0, -1)
    assert agent.direction_to_delta("down") == (0, 1)
    assert agent.direction_to_delta("left") == (-1, 0)
    assert agent.direction_to_delta("right") == (1, 0)

    # Test position update
    agent.update_position((2, 1))
    assert agent.get_position() == (2, 1)
    assert len(memory.positions) == 2  # Initial + update

    print("✅ Agent basic tests passed!")
    return True


def test_agent_fallback():
    """Test Agent fallback decision making."""
    print("Testing Agent fallback decision making...")

    # Temporarily remove API key to force fallback
    original_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

    try:
        memory = AgentMemory()
        agent = Agent(goal=(3, 1), memory=memory, current_position=(1, 1))

        # Create simple grid
        grid = [
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"]
        ]

        # Test decision making
        decision = agent.decide_next_move(grid)
        assert decision is not None
        assert isinstance(decision, tuple)
        assert len(decision) == 2

        # Should move towards goal (right in this case)
        dx, dy = decision
        assert dx == 1 and dy == 0  # Should move right towards goal

        print("✅ Agent fallback tests passed!")

    finally:
        # Restore original API key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key

    return True


def test_integration():
    """Test integration between components."""
    print("Testing component integration...")

    # Create components
    grid = GridWorld(width=6, height=6)
    memory = AgentMemory()
    agent = Agent(goal=(4, 4), memory=memory, current_position=(1, 1))

    # Set up grid
    grid.set_cell_type(4, 4, CellType.GOAL)
    grid.set_cell_type(1, 1, CellType.AGENT)

    # Test a few moves
    for i in range(3):
        grid_data = grid.get_grid_as_list()
        decision = agent.decide_next_move(grid_data)

        if decision:
            dx, dy = decision
            current_pos = agent.get_position()
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if grid.is_valid_position(new_pos[0], new_pos[1]):
                # Clear old position
                grid.set_cell_type(current_pos[0], current_pos[1], CellType.EMPTY)
                # Update agent
                agent.update_position(new_pos)
                # Set new position
                grid.set_cell_type(new_pos[0], new_pos[1], CellType.AGENT)

    # Verify memory was updated
    assert len(memory.positions) >= 3
    assert len(memory.move_history) >= 3

    print("✅ Integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("🧪 Running Self-Evolving Agent Tests\n")

    tests = [
        test_memory,
        test_grid_world,
        test_agent_basic,
        test_agent_fallback,
        test_integration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed!")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
        print()

    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo run the full simulation:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
