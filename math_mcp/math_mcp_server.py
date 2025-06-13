# -*- coding: utf-8 -*-
"""
Math Computation MCP Server - Dynamic Description Loader Version
Provides powerful math computation tools for LLM, using a modular architecture
Tool description information is dynamically loaded from external config files
"""

from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Tuple, Union
import os
from datetime import datetime
import sympy as sp
from typing import Annotated
from pydantic import Field

# Import calculator modules
try:
    from .matrix import MatrixCalculator
    from .mstatistics import StatisticsCalculator
    from .calculus import CalculusCalculator
    from .optimization import OptimizationCalculator
    from .regression import RegressionCalculator
    from .plotting import PlottingCalculator
    from .basic import BasicCalculator

    # Additional extension modules
    from .geometry import GeometryCalculator
    from .number_theory import NumberTheoryCalculator
    from .signal_processing import SignalProcessingCalculator
    from .financial import FinancialCalculator
    from .probability import ProbabilityCalculator
    from .complex_analysis import ComplexAnalysisCalculator
    from .graph_theory import GraphTheoryCalculator
except ImportError:
    from math_mcp.matrix import MatrixCalculator
    from math_mcp.mstatistics import StatisticsCalculator
    from math_mcp.calculus import CalculusCalculator
    from math_mcp.optimization import OptimizationCalculator
    from math_mcp.regression import RegressionCalculator
    from math_mcp.plotting import PlottingCalculator
    from math_mcp.basic import BasicCalculator

    # Additional extension modules
    from math_mcp.geometry import GeometryCalculator
    from math_mcp.number_theory import NumberTheoryCalculator
    from math_mcp.signal_processing import SignalProcessingCalculator
    from math_mcp.financial import FinancialCalculator
    from math_mcp.probability import ProbabilityCalculator
    from math_mcp.complex_analysis import ComplexAnalysisCalculator
    from math_mcp.graph_theory import GraphTheoryCalculator

# Create FastMCP application
mcp = FastMCP("math-calculator")

# Initialize all calculators
matrix_calc = MatrixCalculator()
stats_calc = StatisticsCalculator()
calculus_calc = CalculusCalculator()
optimization_calc = OptimizationCalculator()
regression_calc = RegressionCalculator()
plotting_calc = PlottingCalculator()
basic_calc = BasicCalculator()
# Additional extension calculators
geometry_calc = GeometryCalculator()
number_theory_calc = NumberTheoryCalculator()
signal_calc = SignalProcessingCalculator()
financial_calc = FinancialCalculator()
probability_calc = ProbabilityCalculator()
complex_calc = ComplexAnalysisCalculator()
graph_calc = GraphTheoryCalculator()


# === Basic Numeric Calculation Tools ===
# Basic Arithmetic
@mcp.tool()
def basic_arithmetic(
    operation: Annotated[
        str,
        Field(
            description="Arithmetic operation type. Supports: 'add', 'subtract', 'multiply', 'product', 'divide', 'power', 'modulo', 'modulus', 'factorial', 'gcd', 'lcm', 'sum', 'average'"
        ),
    ],
    numbers: Annotated[
        List[float],
        Field(
            description="List of numbers for the operation. Must contain at least one number"
        ),
    ],
    precision: Annotated[
        Optional[int],
        Field(
            description="Number of decimal places for the result. Range 0-15",
            ge=0,
            le=15,
        ),
    ] = None,
    use_decimal: Annotated[
        bool, Field(description="Whether to use high-precision decimal calculation")
    ] = False,
) -> Dict[str, Any]:
    """
    Brief description: Basic arithmetic operations tool, supporting addition, subtraction, multiplication, division, and high-precision calculations. //
    Examples:
        basic_arithmetic(operation='add', numbers=[1, 2, 3, 4, 5])
        basic_arithmetic(operation='multiply', numbers=[2.5, 3.7], precision=3)
    """
    try:
        return basic_calc.basic_arithmetic_tool(
            operation, numbers, precision, use_decimal
        )
    except Exception as e:
        return {"error": f"Error in basic arithmetic operation: {str(e)}"}


# Mathematical Functions
@mcp.tool()
def mathematical_functions(
    function: Annotated[
        str,
        Field(
            description="Mathematical function type. Supports: 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'log', 'log10', 'ln', 'exp', 'sqrt', 'cbrt', 'abs', 'ceil', 'floor', 'round', 'factorial', 'gamma'"
        ),
    ],
    value: Annotated[float, Field(description="Input value for the function")],
    base: Annotated[
        Optional[float],
        Field(
            description="Base for logarithm. Only for 'log' function, defaults to 10"
        ),
    ] = None,
    precision: Annotated[
        Optional[int],
        Field(
            description="Number of decimal places for the result. Range 0-15",
            ge=0,
            le=15,
        ),
    ] = None,
    angle_unit: Annotated[
        str,
        Field(
            description="Angle unit. Used for trigonometric functions, supports 'radians', 'degrees'"
        ),
    ] = "radians",
) -> Dict[str, Any]:
    """
    Brief description: Mathematical function calculation tool, supporting trigonometric, logarithmic, exponential functions, etc.
    Examples:
        mathematical_functions(function='sin', value=1.57, angle_unit='radians')
        mathematical_functions(function='log', value=100, base=10)
    """
    try:
        return basic_calc.mathematical_functions_tool(
            function, value, base, precision, angle_unit
        )
    except Exception as e:
        return {"error": f"Error in mathematical function calculation: {str(e)}"}


# Number Converter
@mcp.tool()
def number_converter(
    number: Annotated[
        str, Field(description="The number to convert (provided as a string)")
    ],
    from_base: Annotated[
        int, Field(description="Source base. Range 2-36", ge=2, le=36)
    ] = 10,
    to_base: Annotated[
        int, Field(description="Target base. Range 2-36", ge=2, le=36)
    ] = 10,
    operation: Annotated[
        str,
        Field(
            description="Type of conversion operation. Supports: 'convert'(base conversion), 'format'(formatting), 'scientific'(scientific notation), 'fraction'(fraction form)"
        ),
    ] = "convert",
    precision: Annotated[
        Optional[int],
        Field(
            description="Precision. Number of decimal places for scientific/engineering notation",
            ge=0,
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Number format conversion tool, supporting base conversion, scientific notation, etc.
    Examples:
        number_converter(number='255', from_base=10, to_base=16)
        number_converter(number='1010', from_base=2, to_base=10)
    """
    try:
        return basic_calc.number_converter_tool(
            number, from_base, to_base, operation, precision
        )
    except Exception as e:
        return {"error": f"Error in number conversion: {str(e)}"}


# Unit Converter
@mcp.tool()
def unit_converter(
    value: Annotated[float, Field(description="The numerical value to convert")],
    from_unit: Annotated[str, Field(description="Source unit")],
    to_unit: Annotated[str, Field(description="Target unit")],
    unit_type: Annotated[
        str,
        Field(
            description="Unit type. Supports: 'length', 'weight', 'temperature', 'area', 'volume', 'time', 'speed', 'energy'"
        ),
    ],
) -> Dict[str, Any]:
    """
    Brief description: Physical unit conversion tool, supporting length, weight, temperature, etc., unit conversions.
    Examples:
        unit_converter(value=100, from_unit='cm', to_unit='m', unit_type='length')
        unit_converter(value=32, from_unit='fahrenheit', to_unit='celsius', unit_type='temperature')
    """
    try:
        return basic_calc.unit_converter_tool(value, from_unit, to_unit, unit_type)
    except Exception as e:
        return {"error": f"Error in unit conversion: {str(e)}"}


# Precision Calculator
@mcp.tool()
def precision_calculator(
    numbers: Annotated[
        List[float], Field(description="List of numerical values for calculation")
    ],
    operation: Annotated[
        str,
        Field(
            description="Operation type. Supports: 'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'factorial'"
        ),
    ],
    precision_digits: Annotated[
        int, Field(description="Number of precision digits", ge=1)
    ] = 10,
    rounding_mode: Annotated[
        str,
        Field(
            description="Rounding mode. Supports: 'round_half_up', 'round_down', etc."
        ),
    ] = "round_half_up",
) -> Dict[str, Any]:
    """
    Brief description: High-precision numerical calculation tool, supporting arithmetic operations with arbitrary precision.
    Examples:
        precision_calculator(numbers=[1.123, 2.987], operation='add', precision_digits=15)
        precision_calculator(numbers=[2], operation='sqrt', precision_digits=20)
    """
    try:
        return basic_calc.precision_calculator_tool(
            numbers, operation, precision_digits, rounding_mode
        )
    except Exception as e:
        return {"error": f"Error in high precision calculation: {str(e)}"}


# Number Properties
@mcp.tool()
def number_properties(
    number: Annotated[float, Field(description="The number to analyze")],
    analysis_type: Annotated[
        str,
        Field(
            description="Type of analysis. Supports: 'comprehensive'(comprehensive), 'prime'(prime), 'factor'(factor), 'digital'(digital features), 'classification'(classification)"
        ),
    ] = "comprehensive",
) -> Dict[str, Any]:
    """
    Brief description: Numerical property analysis tool, analyzes various mathematical properties of numbers.
    Examples:
        number_properties(number=17, analysis_type='comprehensive')
        number_properties(number=100, analysis_type='factor')
    """
    try:
        return basic_calc.number_properties_tool(number, analysis_type)
    except Exception as e:
        return {"error": f"Error in number property analysis: {str(e)}"}


# === Matrix Calculation Tools ===
@mcp.tool()
def matrix_calculator(
    operation: Annotated[
        str,
        Field(
            description="Matrix operation type. Supports: 'add', 'subtract', 'multiply', 'transpose', 'determinant', 'inverse', 'rank', 'eigenvalues', 'eigenvectors', 'norm', 'trace', 'decomposition', 'solve_system', 'properties'"
        ),
    ],
    matrix_a: Annotated[
        List[List[float]], Field(description="Primary matrix as a 2D list")
    ],
    matrix_b: Annotated[
        Optional[List[List[float]]],
        Field(
            description="Secondary matrix as a 2D list. Required for binary operations"
        ),
    ] = None,
    method: Annotated[
        Optional[str],
        Field(
            description="Method for decomposition. Supports: 'lu', 'qr', 'svd', 'cholesky'"
        ),
    ] = None,
    power: Annotated[
        Optional[int], Field(description="Power for matrix exponentiation", ge=0)
    ] = None,
    property_type: Annotated[
        Optional[str],
        Field(
            description="Type of matrix property to analyze. Supports: 'all', 'basic', 'advanced'"
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Matrix and linear algebra calculation tool, supporting basic operations and advanced analysis.
    Examples:
        matrix_calculator(operation='multiply', matrix_a=[[1,2],[3,4]], matrix_b=[[5,6],[7,8]])
        matrix_calculator(operation='eigenvalues', matrix_a=[[4,2],[1,3]])
    """
    try:
        return matrix_calc.matrix_calculator_tool(
            operation, matrix_a, matrix_b, method, power, property_type
        )
    except Exception as e:
        return {"error": f"Error in matrix calculation: {str(e)}"}


# === Statistics Tools ===
@mcp.tool()
def statistics_analyzer(
    data1: Annotated[List[float], Field(description="Primary dataset for analysis")],
    analysis_type: Annotated[
        str,
        Field(
            description="Type of statistical analysis. Supports: 'descriptive', 'distribution', 'hypothesis_test', 'correlation', 'regression', 'comparison'"
        ),
    ],
    data2: Annotated[
        Optional[List[float]],
        Field(description="Secondary dataset for comparison operations"),
    ] = None,
    test_type: Annotated[
        Optional[str],
        Field(
            description="Statistical test type. Supports: 't_test', 'chi_square', 'anova', 'kolmogorov_smirnov'"
        ),
    ] = None,
    hypothesis_test_type: Annotated[
        Optional[str],
        Field(
            description="Hypothesis test variant. Supports: 'one_sample', 'two_sample', 'paired'"
        ),
    ] = None,
    confidence: Annotated[
        float, Field(description="Confidence level", gt=0, lt=1)
    ] = 0.95,
    distribution_type: Annotated[
        Optional[str],
        Field(
            description="Distribution type for analysis. Supports: 'normal', 'binomial', 'poisson'"
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Comprehensive statistical analysis tool, supporting descriptive statistics, hypothesis testing, and distribution analysis.
    Examples:
        statistics_analyzer(data1=[1,2,3,4,5], analysis_type='descriptive')
        statistics_analyzer(data1=[1,2,3], data2=[4,5,6], analysis_type='comparison')
    """
    try:
        return stats_calc.statistics_analyzer_tool(
            data1,
            analysis_type,
            data2,
            test_type,
            hypothesis_test_type,
            confidence,
            distribution_type,
        )
    except Exception as e:
        return {"error": f"Error in statistical analysis: {str(e)}"}


# === Calculus Tools ===
@mcp.tool()
def calculus_engine(
    expression: Annotated[str, Field(description="Mathematical expression to analyze")],
    operation: Annotated[
        str,
        Field(
            description="Calculus operation type. Supports: 'derivative', 'integral', 'limit', 'series', 'differential_equation', 'optimization', 'taylor_series', 'fourier_series'"
        ),
    ],
    variable: Annotated[str, Field(description="Variable symbol")] = "x",
    variables: Annotated[
        Optional[List[str]],
        Field(description="List of variables for multivariate functions"),
    ] = None,
    limits: Annotated[
        Optional[List[float]],
        Field(description="Integration limits or limit calculation bounds"),
    ] = None,
    point: Annotated[
        Optional[float], Field(description="Point for derivative or limit evaluation")
    ] = None,
    points: Annotated[
        Optional[List[float]],
        Field(description="Multiple points for evaluation"),
    ] = None,
    order: Annotated[
        int, Field(description="Order of derivative or series terms", ge=1)
    ] = 1,
    method: Annotated[
        str,
        Field(
            description="Calculation method. Supports: 'quad', 'dblquad', 'symbolic', 'numerical'"
        ),
    ] = "quad",
    mode: Annotated[
        str, Field(description="Calculation mode. Supports: 'symbolic', 'numerical'")
    ] = "symbolic",
) -> Dict[str, Any]:
    """
    Brief description: Advanced calculus computation engine, supporting derivatives, integrals, limits, series, and differential equations.
    Examples:
        calculus_engine(expression='x**2 + 3*x + 1', operation='derivative', variable='x')
        calculus_engine(expression='sin(x)', operation='integral', variable='x', limits=[0, 3.14159])
    """
    try:
        return calculus_calc.calculus_engine_tool(
            expression,
            operation,
            variable,
            variables,
            limits,
            point,
            points,
            order,
            method,
            mode,
        )
    except Exception as e:
        return {"error": f"Error in calculus computation: {str(e)}"}


# === Optimization Tools ===
@mcp.tool()
def optimization_suite(
    objective_function: Annotated[
        Optional[str],
        Field(description="Objective function expression to optimize"),
    ] = None,
    variables: Annotated[
        Optional[List[str]],
        Field(description="List of optimization variables"),
    ] = None,
    operation: Annotated[
        str,
        Field(
            description="Optimization operation. Supports: 'minimize', 'maximize', 'find_roots', 'linear_programming'"
        ),
    ] = "minimize",
    method: Annotated[
        str,
        Field(
            description="Optimization method. Supports: 'auto', 'nelder_mead', 'powell', 'bfgs', 'lbfgs', 'differential_evolution'"
        ),
    ] = "auto",
    initial_guess: Annotated[
        Optional[List[float]],
        Field(description="Initial guess for optimization variables"),
    ] = None,
    bounds: Annotated[
        Optional[List[Tuple[float, float]]],
        Field(description="Bounds for variables as list of (min, max) tuples"),
    ] = None,
    constraints: Annotated[
        Optional[List[Dict[str, str]]],
        Field(description="Constraint definitions as list of dictionaries"),
    ] = None,
    equation: Annotated[
        Optional[str],
        Field(description="Equation to solve for root finding"),
    ] = None,
    root_method: Annotated[
        str,
        Field(
            description="Root finding method. Supports: 'fsolve', 'brentq', 'newton'"
        ),
    ] = "fsolve",
    lp_c: Annotated[
        Optional[List[float]],
        Field(description="Coefficients for linear programming objective"),
    ] = None,
    lp_A_ub: Annotated[
        Optional[List[List[float]]],
        Field(description="Inequality constraint matrix for linear programming"),
    ] = None,
    lp_b_ub: Annotated[
        Optional[List[float]],
        Field(description="Inequality constraint bounds for linear programming"),
    ] = None,
    lp_A_eq: Annotated[
        Optional[List[List[float]]],
        Field(description="Equality constraint matrix for linear programming"),
    ] = None,
    lp_b_eq: Annotated[
        Optional[List[float]],
        Field(description="Equality constraint bounds for linear programming"),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Professional optimization suite, supporting function optimization, constraint optimization, root finding, and linear programming.
    Examples:
        optimization_suite(objective_function='x**2 + y**2', variables=['x', 'y'], operation='minimize')
        optimization_suite(equation='x**2 - 4', operation='find_roots')
    """
    try:
        return optimization_calc.optimization_suite_tool(
            objective_function or "",
            variables or [],
            operation,
            method,
            initial_guess,
            bounds,
            constraints,
            equation,
            root_method,
            lp_c,
            lp_A_ub,
            lp_b_ub,
            lp_A_eq,
            lp_b_eq,
        )
    except Exception as e:
        return {"error": f"Error in optimization calculation: {str(e)}"}


# === Regression Modeling Tools ===
@mcp.tool()
def regression_modeler(
    operation: Annotated[
        str,
        Field(
            description="Regression operation type. Supports: 'fit', 'predict', 'residual_analysis', 'model_comparison'"
        ),
    ] = "fit",
    x_data: Annotated[
        Optional[List[List[float]]],
        Field(description="Independent variable data as 2D list"),
    ] = None,
    y_data: Annotated[
        Optional[List[float]],
        Field(description="Dependent variable data as 1D list"),
    ] = None,
    model_type: Annotated[
        str,
        Field(
            description="Regression model type. Supports: 'linear', 'polynomial', 'ridge', 'lasso', 'elastic_net', 'logistic'"
        ),
    ] = "linear",
    degree: Annotated[
        int, Field(description="Degree for polynomial regression", ge=1)
    ] = 2,
    alpha: Annotated[float, Field(description="Regularization parameter", ge=0)] = 1.0,
    l1_ratio: Annotated[
        float, Field(description="Elastic Net L1 ratio", ge=0, le=1)
    ] = 0.5,
    cv_folds: Annotated[
        int, Field(description="Number of cross-validation folds", ge=2)
    ] = 5,
    test_size: Annotated[
        float, Field(description="Test set proportion", gt=0, lt=1)
    ] = 0.2,
    y_true: Annotated[
        Optional[List[float]],
        Field(description="True values for residual analysis"),
    ] = None,
    y_pred: Annotated[
        Optional[List[float]],
        Field(description="Predicted values for residual analysis"),
    ] = None,
    models_results: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(description="List of model results for comparison"),
    ] = None,
    training_x: Annotated[
        Optional[List[List[float]]],
        Field(description="Training independent variable data"),
    ] = None,
    training_y: Annotated[
        Optional[List[float]],
        Field(description="Training dependent variable data"),
    ] = None,
    model_params: Annotated[
        Optional[Dict[str, Any]],
        Field(description="Pre-trained model parameters"),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Regression analysis and machine learning modeling tool, supporting various regression algorithms and prediction functions.
    Examples:
        regression_modeler(operation='fit', x_data=[[1], [2], [3]], y_data=[2, 4, 6], model_type='linear')
        regression_modeler(operation='predict', x_data=[[12]], training_x=[[1], [2], [3]], training_y=[2, 4, 6])
    """
    try:
        return regression_calc.regression_modeler_tool(
            operation=operation,
            x_data=x_data,
            y_data=y_data,
            model_type=model_type,
            degree=degree,
            alpha=alpha,
            l1_ratio=l1_ratio,
            cv_folds=cv_folds,
            test_size=test_size,
            y_true=y_true,
            y_pred=y_pred,
            models_results=models_results,
            training_x=training_x,
            training_y=training_y,
            model_params=model_params,
        )
    except Exception as e:
        return {"error": f"Error in regression modeling: {str(e)}"}


# === Expression Evaluator Tools ===
@mcp.tool()
def expression_evaluator(
    expression: Annotated[
        str, Field(description="Mathematical expression to evaluate")
    ],
    variables: Annotated[
        Optional[Dict[str, float]],
        Field(description="Dictionary of variable values, e.g. {'x': 1, 'y': 2}"),
    ] = None,
    mode: Annotated[
        str,
        Field(
            description="Computation mode. Supports: 'evaluate', 'simplify', 'expand', 'factor'"
        ),
    ] = "evaluate",
    output_format: Annotated[
        str,
        Field(
            description="Output format. Supports: 'decimal', 'fraction', 'scientific', 'latex'"
        ),
    ] = "decimal",
) -> Dict[str, Any]:
    """
    Brief description: Mathematical expression evaluation and symbolic computation tool.
    Examples:
        expression_evaluator(expression='2*x + 3*y', variables={'x': 5, 'y': 7})
        expression_evaluator(expression='x**2 + 2*x + 1', mode='factor')
    """
    try:
        expr = sp.sympify(expression)

        if mode == "evaluate" and variables:
            result = expr.subs(variables)
            result_value = float(result) if result.is_number else str(result)
            return {
                "expression": expression,
                "variables": variables,
                "result": result_value,
                "mode": mode,
            }
        elif mode == "simplify":
            simplified = sp.simplify(expr)
            return {
                "expression": expression,
                "simplified": str(simplified),
                "mode": mode,
            }
        elif mode == "expand":
            expanded = sp.expand(expr)
            return {"expression": expression, "expanded": str(expanded), "mode": mode}
        elif mode == "factor":
            factored = sp.factor(expr)
            return {"expression": expression, "factored": str(factored), "mode": mode}
        else:
            return {"expression": expression, "symbolic_form": str(expr), "mode": mode}
    except Exception as e:
        return {"error": f"Error in expression evaluation: {str(e)}"}


# === Plotting Tools ===
# Create and Save Chart
@mcp.tool()
def create_and_save_chart(
    chart_type: Annotated[
        str,
        Field(
            description="Chart type. Supports: 'bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap', 'correlation_matrix', 'multi_series_line'"
        ),
    ],
    data: Annotated[
        Optional[List[float]],
        Field(description="Single series data for bar, pie, histogram, box plots"),
    ] = None,
    x_data: Annotated[
        Optional[List[float]],
        Field(description="X-axis data for line, scatter, multi-series line plots"),
    ] = None,
    y_data: Annotated[
        Optional[List[float]],
        Field(description="Y-axis data for line, scatter plots"),
    ] = None,
    y_data_series: Annotated[
        Optional[List[List[float]]],
        Field(description="Multi-series Y-axis data for multi-series line plots"),
    ] = None,
    series_labels: Annotated[
        Optional[List[str]],
        Field(description="Labels for multi-series plots"),
    ] = None,
    matrix_data: Annotated[
        Optional[List[List[float]]],
        Field(description="Matrix data for heatmaps, correlation matrices"),
    ] = None,
    labels: Annotated[
        Optional[List[str]],
        Field(description="Data labels for bar charts, pie charts"),
    ] = None,
    title: Annotated[str, Field(description="Chart title")] = "Statistical Chart",
    xlabel: Annotated[str, Field(description="X-axis label")] = "X Axis",
    ylabel: Annotated[str, Field(description="Y-axis label")] = "Y Axis",
    filename: Annotated[
        Optional[str],
        Field(description="Save filename without path and extension"),
    ] = None,
    format: Annotated[
        str,
        Field(description="Image format. Supports: 'png', 'jpg', 'svg'"),
    ] = "png",
    colors: Annotated[
        Optional[List[str]],
        Field(description="List of custom colors for chart elements"),
    ] = None,
    figsize: Annotated[
        Optional[Tuple[float, float]],
        Field(description="Figure size as (width, height)"),
    ] = None,
    dpi: Annotated[int, Field(description="Image resolution", ge=50)] = 300,
    style: Annotated[str, Field(description="Chart style")] = "whitegrid",
    timestamp: Annotated[
        bool, Field(description="Whether to add timestamp to filename")
    ] = True,
    show_values: Annotated[
        bool, Field(description="Whether to display value labels on bar charts")
    ] = False,
    horizontal: Annotated[
        bool, Field(description="Whether to create horizontal bar chart")
    ] = False,
    trend_line: Annotated[
        bool, Field(description="Whether to display trend line on scatter plots")
    ] = False,
    trend_line_color: Annotated[
        Optional[str], Field(description="Trend line color")
    ] = None,
    trend_line_equation: Annotated[
        Optional[str], Field(description="Trend line equation text")
    ] = None,
    bins: Annotated[int, Field(description="Number of bins for histogram", ge=1)] = 30,
    annotate: Annotated[
        bool, Field(description="Whether to display numerical annotations on heatmaps")
    ] = True,
    colormap: Annotated[str, Field(description="Colormap for heatmaps")] = "viridis",
    color: Annotated[Optional[str], Field(description="Single color")] = None,
    line_width: Annotated[
        float, Field(description="Line width for line plots", gt=0)
    ] = 2.0,
    line_style: Annotated[str, Field(description="Line style")] = "-",
    marker: Annotated[str, Field(description="Marker style")] = "o",
    marker_size: Annotated[int, Field(description="Marker size", ge=1)] = 6,
    alpha: Annotated[float, Field(description="Transparency level", ge=0, le=1)] = 0.7,
    grid: Annotated[bool, Field(description="Whether to display grid")] = True,
) -> Dict[str, Any]:
    """
    Brief description: Data visualization and chart creation tool, supporting various statistical chart types.
    Examples:
        create_and_save_chart(chart_type='line', x_data=[1,2,3,4], y_data=[1,4,2,3], title='Line Plot')
        create_and_save_chart(chart_type='histogram', data=[1,2,2,3,3,3,4,4,5], filename='histogram_plot')
    """
    try:
        return plotting_calc.statistical_plotter_tool(
            chart_type=chart_type,
            data=data,
            x_data=x_data,
            y_data=y_data,
            y_data_series=y_data_series,
            series_labels=series_labels,
            matrix_data=matrix_data,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            format=format,
            colors=colors,
            figsize=figsize,
            dpi=dpi,
            style=style,
            show_values=show_values,
            horizontal=horizontal,
            trend_line=trend_line,
            trend_line_color=trend_line_color,
            trend_line_equation=trend_line_equation,
            bins=bins,
            annotate=annotate,
            colormap=colormap,
            color=color,
            line_width=line_width,
            line_style=line_style,
            marker=marker,
            marker_size=marker_size,
            alpha=alpha,
            grid=grid,
        )
    except Exception as e:
        return {"error": f"Error in statistical chart plotting: {str(e)}"}


# Plot Function Curve
@mcp.tool()
def plot_function_curve(
    function_expression: Annotated[
        str,
        Field(
            description="Function expression with common math functions like sin, cos, exp, log, sqrt"
        ),
    ],
    variable: Annotated[str, Field(description="Independent variable name")] = "x",
    x_range: Annotated[
        Tuple[float, float],
        Field(description="X-axis range as (min_value, max_value)"),
    ] = (-10, 10),
    num_points: Annotated[
        int, Field(description="Number of plotting points", ge=10)
    ] = 1000,
    title: Annotated[str, Field(description="Chart title")] = "Function Graph",
    xlabel: Annotated[str, Field(description="X-axis label")] = "X Axis",
    ylabel: Annotated[str, Field(description="Y-axis label")] = "Y Axis",
    filename: Annotated[
        Optional[str],
        Field(description="Save filename without path and extension"),
    ] = None,
    format: Annotated[
        str, Field(description="Image format. Supports: 'png', 'jpg', 'svg'")
    ] = "png",
    figsize: Annotated[
        Tuple[float, float], Field(description="Figure size as (width, height)")
    ] = (10, 6),
    dpi: Annotated[int, Field(description="Image resolution", ge=50)] = 300,
    color: Annotated[str, Field(description="Function curve color")] = "blue",
    line_width: Annotated[float, Field(description="Line width", gt=0)] = 2.0,
    grid: Annotated[bool, Field(description="Whether to display grid")] = True,
    grid_alpha: Annotated[
        float, Field(description="Grid transparency", ge=0, le=1)
    ] = 0.3,
    derivative_order: Annotated[
        Optional[int],
        Field(description="Derivative order for plotting n-th derivative curve"),
    ] = None,
    show_critical_points: Annotated[
        bool,
        Field(description="Whether to show critical points (extrema)"),
    ] = False,
    show_equation: Annotated[
        bool, Field(description="Whether to display function equation on plot")
    ] = True,
    equation_position: Annotated[
        str,
        Field(
            description="Equation display position. Supports: 'upper right', 'upper left', 'lower right', 'lower left'"
        ),
    ] = "upper right",
    alpha: Annotated[float, Field(description="Line transparency", ge=0, le=1)] = 1.0,
    line_style: Annotated[str, Field(description="Line style")] = "-",
    marker: Annotated[str, Field(description="Data point marker")] = "",
    marker_size: Annotated[int, Field(description="Marker size", ge=1)] = 6,
) -> Dict[str, Any]:
    """
    Brief description: Mathematical function curve plotting tool, supporting function graph visualization and derivative analysis.
    Examples:
        plot_function_curve(function_expression='x**2 + 2*x + 1')
        plot_function_curve(function_expression='sin(x)', x_range=(-6.28, 6.28), filename='sine_wave')
    """
    try:
        return plotting_calc.plot_function_tool(
            function_expression=function_expression,
            variable=variable,
            x_range=x_range,
            num_points=num_points,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            format=format,
            figsize=figsize,
            dpi=dpi,
            color=color,
            line_width=line_width,
            grid=grid,
            grid_alpha=grid_alpha,
            derivative_order=derivative_order,
            show_critical_points=show_critical_points,
            show_equation=show_equation,
            equation_position=equation_position,
            alpha=alpha,
            line_style=line_style,
            marker=marker,
            marker_size=marker_size,
        )
    except Exception as e:
        return {"error": f"Error in function plotting: {str(e)}"}


# === Geometry Tools ===
@mcp.tool()
def geometry_calculator(
    shape_type: Annotated[
        str,
        Field(
            description="Type of geometric shape. Supports: 'circle', 'triangle', 'rectangle', 'polygon', 'ellipse', 'sphere', 'cube', 'cylinder', 'cone', 'pyramid'"
        ),
    ],
    operation: Annotated[
        str,
        Field(
            description="Geometric operation. Supports: 'area', 'volume', 'surface_area', 'circumference', 'perimeter', 'properties', 'distance', 'angle'"
        ),
    ],
    dimensions: Annotated[
        Optional[Dict[str, float]],
        Field(
            description="Dictionary of dimension parameters, e.g. {'radius': 5}, {'length': 10, 'width': 5}"
        ),
    ] = None,
    points: Annotated[
        Optional[List[List[float]]],
        Field(
            description="List of coordinate points for coordinate-based calculations"
        ),
    ] = None,
    precision: Annotated[
        Optional[int],
        Field(description="Number of decimal places for results", ge=0, le=15),
    ] = None,
    unit: Annotated[str, Field(description="Measurement unit identifier")] = "default",
) -> Dict[str, Any]:
    """
    Brief description: Powerful geometry calculation tool, supporting plane geometry, solid geometry, and analytical geometry calculations.
    Examples:
        geometry_calculator(shape_type='circle', operation='properties', dimensions={'radius': 5})
        geometry_calculator(shape_type='triangle', operation='area', points=[[0,0], [3,0], [0,4]])
    """
    try:
        return geometry_calc.geometry_calculator_tool(
            shape_type, operation, dimensions, points, precision, unit
        )
    except Exception as e:
        return {"error": f"Error in geometry calculation: {str(e)}"}


# === Number Theory Tools ===
@mcp.tool()
def number_theory_calculator(
    operation: Annotated[
        str,
        Field(
            description="Number theory operation. Supports: 'prime_factorization', 'prime_test', 'generate_primes', 'modular_arithmetic', 'extended_gcd', 'euler_totient', 'fibonacci'"
        ),
    ],
    number: Annotated[
        Optional[int],
        Field(description="Primary operand, must be a positive integer"),
    ] = None,
    numbers: Annotated[
        Optional[List[int]],
        Field(description="List of numbers for operations requiring multiple numbers"),
    ] = None,
    modulus: Annotated[
        Optional[int],
        Field(description="Modulus for modular arithmetic, must be positive"),
    ] = None,
    base: Annotated[
        Optional[int], Field(description="Base for modular exponentiation")
    ] = None,
    exponent: Annotated[
        Optional[int],
        Field(description="Exponent for modular exponentiation, must be non-negative"),
    ] = None,
    limit: Annotated[
        Optional[int],
        Field(description="Limit value for prime generation or fibonacci sequence"),
    ] = None,
    precision: Annotated[
        Optional[int],
        Field(
            description="Number of terms for continued fraction expansion", ge=1, le=50
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Advanced number theory calculation tool, supporting prime testing, factorization, modular arithmetic, etc.
    Examples:
        number_theory_calculator(operation='prime_factorization', number=60)
        number_theory_calculator(operation='prime_test', number=97)
    """
    try:
        return number_theory_calc.number_theory_tool(
            operation, number, numbers, modulus, base, exponent, limit, precision
        )
    except Exception as e:
        return {"error": f"Error in number theory calculation: {str(e)}"}


# === Signal Processing Tools ===
@mcp.tool()
def signal_processing_calculator(
    operation: Annotated[
        str,
        Field(
            description="Signal processing operation. Supports: 'fft', 'generate_signal', 'filter', 'windowing', 'autocorrelation', 'spectral_analysis', 'modulation'"
        ),
    ],
    signal: Annotated[
        Optional[List[float]],
        Field(description="Input signal data for small datasets (<1000 points)"),
    ] = None,
    signal_file: Annotated[
        Optional[str],
        Field(description="Signal file path (.json format) for large data processing"),
    ] = None,
    sampling_rate: Annotated[
        Optional[float],
        Field(description="Sampling rate in Hz, must be positive"),
    ] = None,
    frequency: Annotated[
        Optional[float],
        Field(description="Signal frequency in Hz for generation/modulation"),
    ] = None,
    filter_type: Annotated[
        Optional[str],
        Field(
            description="Filter type. Supports: 'lowpass', 'highpass', 'bandpass', 'moving_average'"
        ),
    ] = None,
    cutoff_freq: Annotated[
        Optional[float],
        Field(description="Cutoff frequency in Hz for filtering"),
    ] = None,
    window_size: Annotated[
        Optional[int],
        Field(
            description="Window size for windowing/spectral analysis", ge=3, le=10000
        ),
    ] = None,
    overlap: Annotated[
        Optional[float],
        Field(description="Window overlap ratio for spectral analysis", ge=0, le=1),
    ] = None,
    order: Annotated[
        Optional[int], Field(description="Filter order", ge=1, le=20)
    ] = None,
    duration: Annotated[
        Optional[float],
        Field(description="Signal duration in seconds for generation"),
    ] = None,
    noise_level: Annotated[
        Optional[float], Field(description="Noise amplitude for signal generation")
    ] = None,
    signal_type: Annotated[
        Optional[str],
        Field(
            description="Signal type for generation. Supports: 'sine', 'cosine', 'square', 'sawtooth', 'white_noise'"
        ),
    ] = None,
    output_filename: Annotated[
        Optional[str], Field(description="Output filename without path")
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Professional digital signal processing tool, supporting FFT, filtering, modulation/demodulation, etc.
    Examples:
        signal_processing_calculator(operation='generate_signal', signal_type='sine', frequency=10, sampling_rate=1000, duration=1)
        signal_processing_calculator(operation='fft', signal=[1,2,3,4,5,6,7,8], sampling_rate=8)
    """
    try:
        return signal_calc.signal_processing_tool(
            operation=operation,
            signal=signal,
            signal_file=signal_file,
            sampling_rate=sampling_rate,
            frequency=frequency,
            filter_type=filter_type,
            cutoff_freq=cutoff_freq,
            window_size=window_size,
            overlap=overlap,
            order=order,
            duration=duration,
            noise_level=noise_level,
            signal_type=signal_type,
            output_filename=output_filename,
        )
    except Exception as e:
        return {"error": f"Error in signal processing: {str(e)}"}


# === Financial Tools ===
@mcp.tool()
def financial_calculator(
    operation: Annotated[
        str,
        Field(
            description="Financial calculation operation. Supports: 'compound_interest', 'simple_interest', 'present_value', 'future_value', 'annuity', 'npv', 'irr', 'loan_payment'"
        ),
    ],
    principal: Annotated[
        Optional[float],
        Field(description="Principal or initial investment amount, must be positive"),
    ] = None,
    rate: Annotated[
        Optional[float],
        Field(description="Interest rate per period (decimal format)"),
    ] = None,
    time: Annotated[
        Optional[int],
        Field(description="Time period in years or number of periods"),
    ] = None,
    cash_flows: Annotated[
        Optional[List[float]],
        Field(description="Sequence of cash flows for NPV/IRR calculation"),
    ] = None,
    initial_investment: Annotated[
        Optional[float], Field(description="Initial investment amount")
    ] = None,
    payment: Annotated[
        Optional[float], Field(description="Periodic payment amount")
    ] = None,
    periods: Annotated[
        Optional[int], Field(description="Number of payment periods", ge=1)
    ] = None,
    future_value: Annotated[Optional[float], Field(description="Future value")] = None,
    present_value: Annotated[
        Optional[float], Field(description="Present value")
    ] = None,
    annual_rate: Annotated[
        Optional[float], Field(description="Annualized interest rate (decimal)")
    ] = None,
    payments_per_year: Annotated[
        int, Field(description="Number of payments per year", ge=1)
    ] = 12,
    risk_free_rate: Annotated[
        Optional[float], Field(description="Risk-free rate for Sharpe ratio")
    ] = None,
    returns: Annotated[
        Optional[List[float]],
        Field(description="Sequence of returns for portfolio analysis"),
    ] = None,
    prices: Annotated[
        Optional[List[float]], Field(description="Sequence of asset prices")
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Professional financial mathematics calculation tool, supporting compound interest, investment analysis, risk assessment, etc.
    Examples:
        financial_calculator(operation='compound_interest', principal=1000, rate=0.05, time=10)
        financial_calculator(operation='npv', cash_flows=[-1000, 300, 400, 500], rate=0.1)
    """
    try:
        return financial_calc.financial_calculator_tool(
            operation,
            principal,
            rate,
            time,
            cash_flows,
            initial_investment,
            payment,
            periods,
            future_value,
            present_value,
            annual_rate,
            payments_per_year,
            risk_free_rate,
            returns,
            prices,
        )
    except Exception as e:
        return {"error": f"Error in financial calculation: {str(e)}"}


# === Probability Tools ===
@mcp.tool()
def probability_calculator(
    operation: Annotated[
        str,
        Field(
            description="Probability/statistics operation. Supports: 'probability_mass', 'cumulative_distribution', 'random_sampling', 'bayes_theorem', 'hypothesis_test'"
        ),
    ],
    distribution: Annotated[
        Optional[str],
        Field(
            description="Probability distribution. Supports: 'normal', 'binomial', 'poisson', 'uniform', 'exponential'"
        ),
    ] = None,
    parameters: Annotated[
        Optional[Dict[str, float]],
        Field(description="Distribution parameters, e.g. {'mean': 0, 'std': 1}"),
    ] = None,
    x_value: Annotated[
        Optional[float], Field(description="Single value for probability calculation")
    ] = None,
    x_values: Annotated[
        Optional[List[float]], Field(description="Multiple values for calculation")
    ] = None,
    probability: Annotated[
        Optional[float], Field(description="Probability value for inverse calculations")
    ] = None,
    n_samples: Annotated[
        Optional[int], Field(description="Number of samples for random sampling", ge=1)
    ] = None,
    events: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(description="List of events for Bayesian analysis"),
    ] = None,
    data: Annotated[
        Optional[List[float]], Field(description="Data for statistical testing")
    ] = None,
) -> Dict[str, Any]:
    """
    Brief description: Probability and statistics calculation tool, supporting probability distributions, hypothesis testing, Bayesian analysis, etc.

    Examples:
        probability_calculator(operation='probability_mass', distribution='normal', parameters={'mu':0,'sigma':1}, x_value=1.96)
        probability_calculator(operation='cumulative_distribution', distribution='normal', parameters={'mu':20,'sigma':3}, x_value=25)
        probability_calculator(operation='random_sampling', distribution='binomial', parameters={'n':10,'p':0.3}, n_samples=100)
    """
    try:
        return probability_calc.probability_calculator_tool(
            operation,
            distribution,
            parameters,
            x_value,
            x_values,
            probability,
            n_samples,
            events,
            data,
        )
    except Exception as e:
        return {"error": f"Error in probability calculation: {str(e)}"}


# === Complex Analysis Tools ===
@mcp.tool()
def complex_analysis_suite(
    operation: Annotated[
        str,
        Field(
            description="Complex analysis operation. Supports: 'arithmetic', 'complex_functions', 'contour_integration', 'residue_calculation', 'series_expansion', 'visualization'"
        ),
    ],
    complex_number: Annotated[
        Optional[Union[str, complex]],
        Field(description="Single complex number (string or complex type)"),
    ] = None,
    complex_numbers: Annotated[
        Optional[List[Union[str, complex]]],
        Field(description="List of complex numbers for operations"),
    ] = None,
    function_expression: Annotated[
        Optional[str], Field(description="Complex function expression")
    ] = None,
    variable: Annotated[str, Field(description="Complex variable symbol")] = "z",
    contour_points: Annotated[
        Optional[List[List[float]]],
        Field(description="Contour points for integration"),
    ] = None,
    singularities: Annotated[
        Optional[List[Union[str, complex]]],
        Field(description="List of function singularities"),
    ] = None,
    center: Annotated[
        Optional[Union[str, complex]], Field(description="Center point for expansion")
    ] = None,
    radius: Annotated[
        Optional[float], Field(description="Radius for convergence", gt=0)
    ] = None,
    order: Annotated[
        Optional[int], Field(description="Order of expansion or pole", ge=1)
    ] = None,
    x_range: Annotated[
        Tuple[float, float], Field(description="Real axis range for visualization")
    ] = (-5, 5),
    y_range: Annotated[
        Tuple[float, float], Field(description="Imaginary axis range for visualization")
    ] = (-5, 5),
    resolution: Annotated[
        int, Field(description="Plot resolution", ge=50, le=2000)
    ] = 500,
    colormap: Annotated[str, Field(description="Colormap for visualization")] = "hsv",
    filename: Annotated[
        Optional[str], Field(description="Output filename for plots")
    ] = None,
    plot_type: Annotated[
        str,
        Field(
            description="Visualization type. Supports: 'domain_coloring', 'contour_plot', 'surface_plot'"
        ),
    ] = "domain_coloring",
    series_terms: Annotated[
        int, Field(description="Number of series terms", ge=1, le=50)
    ] = 10,
    branch_cut: Annotated[
        Optional[str], Field(description="Branch cut specification")
    ] = None,
    method: Annotated[str, Field(description="Calculation method")] = "auto",
) -> Dict[str, Any]:
    """
    Brief description:
        Powerful complex analysis and complex function tool, supporting complex number form conversion, residue calculation, analytic continuation, complex plane visualization, and other advanced features.
    Examples:
        complex_analysis_suite(operation='convert_form', complex_number='3+4i')
        complex_analysis_suite(operation='function_evaluation', function_expression='z**2 + 1', complex_number='1+i')
        complex_analysis_suite(operation='residue_calculation', function_expression='1/(z**2 + 1)', singularities=['i', '-i'])
    """
    try:
        return complex_calc.complex_analysis_suite_tool(
            operation,
            complex_number,
            complex_numbers,
            function_expression,
            variable,
            contour_points,
            singularities,
            center,
            radius,
            order,
            x_range,
            y_range,
            resolution,
            colormap,
            filename,
            plot_type,
            series_terms,
            branch_cut,
            method,
        )
    except Exception as e:
        return {"error": f"Error in complex analysis: {str(e)}"}


# === Graph Theory Tools ===
@mcp.tool()
def graph_theory_suite(
    operation: Annotated[
        str,
        Field(
            description="Graph theory operation type. Supports: 'shortest_path', 'all_pairs_shortest_path', 'maximum_flow', 'connectivity_analysis', 'centrality_analysis', 'community_detection', 'spectral_analysis', 'graph_properties', 'minimum_spanning_tree', 'graph_coloring', 'clique_analysis', 'graph_visualization', 'graph_comparison', 'graph_generation'"
        ),
    ],
    graph_data: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="Graph data dictionary. Format: {'nodes': [list of nodes], 'edges': [list of edges]}"
        ),
    ] = None,
    adjacency_matrix: Annotated[
        Optional[List[List[Union[int, float]]]],
        Field(
            description="Adjacency matrix. Square matrix, elements represent connection weights between nodes"
        ),
    ] = None,
    edge_list: Annotated[
        Optional[List[List[Union[int, str]]]],
        Field(
            description="Edge list. Format: [[source_node, target_node], ...] or with weights"
        ),
    ] = None,
    node_list: Annotated[
        Optional[List[Union[int, str]]],
        Field(description="Node list. Used to specify node identifiers"),
    ] = None,
    source_node: Annotated[
        Optional[Union[int, str]],
        Field(description="Source node for algorithms like shortest path"),
    ] = None,
    target_node: Annotated[
        Optional[Union[int, str]],
        Field(description="Target node for algorithms like shortest path"),
    ] = None,
    weight_attribute: Annotated[
        str, Field(description="Weight attribute name for weighted graph algorithms")
    ] = "weight",
    directed: Annotated[
        bool, Field(description="Whether it is a directed graph")
    ] = False,
    algorithm: Annotated[
        str,
        Field(
            description="Algorithm type. E.g., 'dijkstra', 'bellman_ford' for shortest path"
        ),
    ] = "auto",
    k_value: Annotated[
        Optional[int],
        Field(description="K value for K-means clustering, K-core analysis"),
    ] = None,
    threshold: Annotated[
        Optional[float],
        Field(description="Threshold value for filtering edges or nodes"),
    ] = None,
    layout: Annotated[
        str,
        Field(
            description="Graph layout algorithm. Supports: 'spring', 'circular', 'random'"
        ),
    ] = "spring",
    filename: Annotated[
        Optional[str], Field(description="Save filename without path")
    ] = None,
    node_colors: Annotated[
        Optional[List[str]], Field(description="List of node colors for visualization")
    ] = None,
    edge_colors: Annotated[
        Optional[List[str]], Field(description="List of edge colors for visualization")
    ] = None,
    node_sizes: Annotated[
        Optional[List[int]], Field(description="List of node sizes for visualization")
    ] = None,
    show_labels: Annotated[
        bool, Field(description="Whether to show node labels in visualization")
    ] = True,
    figsize: Annotated[
        Tuple[float, float], Field(description="Figure size for visualization")
    ] = (10, 8),
) -> Dict[str, Any]:
    """
    Brief description: Professional graph theory analysis tool, supporting shortest path, maximum flow, connectivity analysis, centrality calculation, community detection, spectral analysis, and other comprehensive graph theory functions.

    Examples:
        graph_theory_suite(operation='shortest_path', edge_list=[[1,2], [2,3], [1,3]], source_node=1, target_node=3)
        graph_theory_suite(operation='centrality_analysis', graph_data={'nodes': [1,2,3], 'edges': [[1,2], [2,3]]})
        graph_theory_suite(operation='graph_visualization', adjacency_matrix=[[0,1,1],[1,0,1],[1,1,0]], filename='graph_plot')
    """
    try:
        return graph_calc.graph_theory_suite_tool(
            operation,
            graph_data,
            adjacency_matrix,
            edge_list,
            node_list,
            source_node,
            target_node,
            weight_attribute,
            directed,
            algorithm,
            k_value,
            threshold,
            layout,
            filename,
            node_colors,
            edge_colors,
            node_sizes,
            show_labels,
            figsize,
        )
    except Exception as e:
        return {"error": f"Error in graph theory analysis: {str(e)}"}


# === Cleanup Resources ===
@mcp.tool()
def cleanup_resources() -> Dict[str, Any]:
    """
    Brief description: Deletes files generated in OUTPUT_PATH (or default temporary directory) and performs basic resource cleanup. Call only when the user explicitly indicates deletion of temporary or output files.

    Examples:
        cleanup_resources()
    """
    try:
        import gc
        import matplotlib.pyplot as plt

        # close all matplotlib figures
        plt.close("all")

        # get output directory
        try:
            from .file_utils import get_output_path
        except ImportError:
            from math_mcp.file_utils import get_output_path

        output_dir = get_output_path()

        removed_files = []
        if os.path.isdir(output_dir):
            for root, _, files in os.walk(output_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                    except Exception:
                        # skip files that cannot be deleted
                        pass

        gc.collect()

        return {
            "success": True,
            "message": "Cleanup finished",
            "deleted_files": len(removed_files),
            "actions": [
                "Closed matplotlib figures",
                "Garbage collection executed",
                f"Removed {len(removed_files)} files from {output_dir}",
            ],
        }
    except Exception as e:
        return {"error": f"Error in resource cleanup: {str(e)}"}


# MCP Server startup function
def main():
    """Start MCP server"""
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server runtime error: {e}")
    finally:
        try:
            import matplotlib.pyplot as plt
            import gc

            plt.close("all")
            gc.collect()
            print("Server safely shut down")
        except Exception as e:
            print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()
