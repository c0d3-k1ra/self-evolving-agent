# Self-Evolving Agent

A Python project that implements an intelligent agent capable of making LLM-based decisions in a grid world environment.

## Features

- **LLM-Based Decision Making**: Agent uses OpenAI API (or compatible endpoints) to make intelligent navigation decisions
- **Memory System**: Comprehensive logging of positions, moves, thoughts, and reasoning
- **Visual Interface**: Real-time pygame visualization of the agent navigating the grid world
- **Flexible Configuration**: Support for custom API endpoints and models via environment variables

## Project Structure

```
self-evolving-agent/
├── main.py                    # Main execution script with pygame loop
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── agent/
│   ├── __init__.py
│   ├── memory.py             # AgentMemory class for tracking history
│   └── agent.py              # Agent class with LLM decision making
└── world/
    ├── __init__.py
    └── grid.py               # GridWorld environment
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

**Option A: Using .env file (Recommended)**
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=http://localhost:8000/v1
```

**Option B: Using shell environment variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_MODEL="gpt-4"  # Optional, default: gpt-3.5-turbo
export OPENAI_BASE_URL="http://localhost:8000/v1"  # Optional, for custom endpoints
```

### 3. Run the Simulation

```bash
python main.py
```

## Usage

### Controls
- **SPACE**: Pause/Resume simulation
- **R**: Reset simulation with new random positions
- **Q**: Quit
- **UP/DOWN**: Adjust simulation speed

### What You'll See
- **White cells**: Empty spaces
- **Gray cells**: Walls (obstacles)
- **Green cell**: Goal position
- **Blue cell**: Agent position
- **Bottom panel**: Real-time information including position, moves, and agent thoughts

## How It Works

1. **Agent Initialization**: Agent starts at a random position with a random goal
2. **LLM Decision Making**: Each move, the agent:
   - Analyzes current position, goal, and grid layout
   - Sends context to LLM asking for next move direction
   - Receives JSON response with move and reasoning
   - Validates and executes the move
3. **Memory Logging**: All decisions, moves, and thoughts are logged
4. **Goal Achievement**: When goal is reached, agent resets with new positions

## Configuration Examples

### Standard OpenAI API
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4"
python main.py
```

### Local LLM (e.g., Ollama with OpenAI compatibility)
```bash
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama2"
python main.py
```

### LiteLLM Proxy
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.litellm.ai/v1"
export OPENAI_MODEL="claude-3-sonnet"
python main.py
```

## Fallback Mode

If no API key is provided, the agent will use a simple fallback navigation algorithm that moves directly toward the goal. This allows you to test the system without LLM integration.

## Extending the System

The modular design makes it easy to extend:

- **Add new tools**: Extend the `AgentMemory` class with new logging methods
- **Custom decision logic**: Modify the `Agent.decide_next_move()` method
- **Different environments**: Create new world types by extending `GridWorld`
- **Enhanced visualization**: Modify the rendering in `main.py`

## Requirements

- Python 3.7+
- pygame 2.5.2
- openai library
- OpenAI API key (or compatible endpoint)
