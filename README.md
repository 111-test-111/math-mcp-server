## Math MCP Server

A Python-based mathematical computation MCP server, providing a suite of mathematical computation tools and plotting utilities.

### Run mcp server
```Python
uvx math-mcp # using uvx

# Or run the .py file directly (dependencies must be installed manually)
python math_mcp/math_mcp_server.py
```

### Project Structure
```
math_mcp/
├── __init__.py                      # Package initialization
├── __main__.py                      # CLI entry point
├── math_mcp_server.py               # Main server file (MCP tool registration)
├── description_loader.py            # Tool description loader
├── tool_descriptions.py             # Tool description configuration
├── file_utils.py                    # File path utilities
# Core computation modules
├── basic.py                         # Basic math computation
├── matrix.py                        # Matrix computations
├── mstatistics.py                   # Statistical analysis (avoid conflicts)
├── calculus.py                      # Calculus
├── optimization.py                  # Optimization algorithms
├── regression.py                    # Regression analysis
├── plotting.py                      # Data visualization
├── geometry.py                      # Geometric computations
├── number_theory.py                 # Number theory
├── complex_analysis.py              # Complex analysis
├── probability.py                   # Probability and statistics
# Extended specialized modules
├── signal_processing.py             # Signal processing
├── financial.py                     # Financial mathematics
└── graph_theory.py                  # Graph theory analysis
```

### Configuration in Claude Desktop

Add the following configuration to your Claude Desktop config file:
```json
{
    "mcpServers": {
        "math-calculator": {
            "command": "uvx",
            "args": ["math-mcp"],
            "env": {
                "OUTPUT_PATH": "path/to/plot_output",
                "FONT_PATH": "path/to/font"
            }
        }
    }
}
```

Or start the server directly:
```json
{
    "mcpServers": {
        "math-calculator-local": {
            "command": "path/to/python_interpreter",
            "args": [
                "path/to/math_mcp_server.py"
            ],
            "env": {
                "OUTPUT_PATH": "path/to/output",
                "FONT_PATH": "path/to/font"
            }
        }
    }
}
```