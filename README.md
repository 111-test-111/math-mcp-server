## Math MCP Server

这是一个数学计算MCP（Model Context Protocol）服务器，提供了一些数学计算工具集和绘图工具

### 功能特性

- **矩阵计算**: 基本运算、矩阵分解、特征值、SVD等
- **统计分析**: 描述性统计、假设检验、分布分析等
- **微积分**: 导数、积分、极限、泰勒级数等
- **优化算法**: 函数优化、线性规划、约束优化等
- **回归分析**: 线性回归、多项式回归、正则化回归等
- **数据可视化**: 统计图表、函数绘图等

### 使用uvx运行

```bash
uvx math-mcp
```

### 项目结构

```
math_mcp/
├── __init__.py                    # 包初始化文件
├── __main__.py                    # CLI入口点
├── math_server.py                 # 主服务器文件（MCP工具注册）
├── matrix_calculator.py           # 矩阵计算模块
├── statistics_calculator.py       # 统计分析模块
├── calculus_calculator.py         # 微积分计算模块
├── optimization_calculator.py     # 优化算法模块
├── regression_calculator.py       # 回归分析模块
└── plotting_calculator.py         # 统计绘图模块
```

### 在Claude Desktop中配置

将以下配置添加到Claude Desktop配置文件中：

```json
{
  "mcpServers": {
    "math-calculator": {
      "command": "uvx",
      "args": ["math-mcp"],
      "env": {
        "OUTPUT_PATH": "path/to/plot_output",
        "FONT_PATH": "path/to/plot_font"
      }
    }
  }
}
```