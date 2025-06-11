"""
Math MCP Server - 命令行入口点
可以通过 python -m math_mcp 或 uvx math-mcp 启动
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径，确保可以导入模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from .math_server import main

if __name__ == "__main__":
    main()
