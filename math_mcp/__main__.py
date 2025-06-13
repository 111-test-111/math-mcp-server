"""
Math MCP Server - Command Line Entry Point
Can be started with python -m math_mcp or uvx math-mcp
"""

import sys
from pathlib import Path

# Add current directory to Python path to ensure modules can be imported
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from .math_mcp_server import main

if __name__ == "__main__":
    main()
