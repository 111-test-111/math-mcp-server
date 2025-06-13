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

# Import calculator modules
try:
    from .description_loader import apply_description
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
    from math_mcp.description_loader import apply_description
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
@apply_description("basic_arithmetic")
def basic_arithmetic(
    operation: str,
    numbers: List[float],
    precision: Optional[int] = None,
    use_decimal: bool = False,
) -> Dict[str, Any]:
    try:
        return basic_calc.basic_arithmetic_tool(
            operation, numbers, precision, use_decimal
        )
    except Exception as e:
        return {"error": f"Error in basic arithmetic operation: {str(e)}"}


# Mathematical Functions
@mcp.tool()
@apply_description("mathematical_functions")
def mathematical_functions(
    function: str,
    value: float,
    base: Optional[float] = None,
    precision: Optional[int] = None,
    angle_unit: str = "radians",
) -> Dict[str, Any]:
    try:
        return basic_calc.mathematical_functions_tool(
            function, value, base, precision, angle_unit
        )
    except Exception as e:
        return {"error": f"Error in mathematical function calculation: {str(e)}"}


# Number Converter
@mcp.tool()
@apply_description("number_converter")
def number_converter(
    number: str,
    from_base: int = 10,
    to_base: int = 10,
    operation: str = "convert",
    precision: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        return basic_calc.number_converter_tool(
            number, from_base, to_base, operation, precision
        )
    except Exception as e:
        return {"error": f"Error in number conversion: {str(e)}"}


# Unit Converter
@mcp.tool()
@apply_description("unit_converter")
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str,
    unit_type: str,
) -> Dict[str, Any]:
    try:
        return basic_calc.unit_converter_tool(value, from_unit, to_unit, unit_type)
    except Exception as e:
        return {"error": f"Error in unit conversion: {str(e)}"}


# Precision Calculator
@mcp.tool()
@apply_description("precision_calculator")
def precision_calculator(
    numbers: List[float],
    operation: str,
    precision_digits: int = 10,
    rounding_mode: str = "round_half_up",
) -> Dict[str, Any]:
    try:
        return basic_calc.precision_calculator_tool(
            numbers, operation, precision_digits, rounding_mode
        )
    except Exception as e:
        return {"error": f"Error in high precision calculation: {str(e)}"}


# Number Properties
@mcp.tool()
@apply_description("number_properties")
def number_properties(
    number: float,
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    try:
        return basic_calc.number_properties_tool(number, analysis_type)
    except Exception as e:
        return {"error": f"Error in number property analysis: {str(e)}"}


# === Matrix Calculation Tools ===
@mcp.tool()
@apply_description("matrix_calculator")
def matrix_calculator(
    operation: str,
    matrix_a: List[List[float]],
    matrix_b: Optional[List[List[float]]] = None,
    method: Optional[str] = None,
    power: Optional[int] = None,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        return matrix_calc.matrix_calculator_tool(
            operation, matrix_a, matrix_b, method, power, property_type
        )
    except Exception as e:
        return {"error": f"Error in matrix calculation: {str(e)}"}


# === Statistical Analysis Tools ===
@mcp.tool()
@apply_description("statistics_analyzer")
def statistics_analyzer(
    data1: List[float],
    analysis_type: str,
    data2: Optional[List[float]] = None,
    test_type: Optional[str] = None,
    hypothesis_test_type: Optional[str] = None,
    confidence: float = 0.95,
    distribution_type: Optional[str] = None,
) -> Dict[str, Any]:
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
@apply_description("calculus_engine")
def calculus_engine(
    expression: str,
    operation: str,
    variable: str = "x",
    variables: Optional[List[str]] = None,
    limits: Optional[List[float]] = None,
    point: Optional[float] = None,
    points: Optional[List[float]] = None,
    order: int = 1,
    method: str = "quad",
    mode: str = "symbolic",
) -> Dict[str, Any]:
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
        return {"error": f"Error in calculus calculation: {str(e)}"}


# === Optimization Tools ===
@mcp.tool()
@apply_description("optimization_suite")
def optimization_suite(
    objective_function: Optional[str] = None,
    variables: Optional[List[str]] = None,
    operation: str = "minimize",
    method: str = "auto",
    initial_guess: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    constraints: Optional[List[Dict[str, str]]] = None,
    equation: Optional[str] = None,
    root_method: str = "fsolve",
    lp_c: Optional[List[float]] = None,
    lp_A_ub: Optional[List[List[float]]] = None,
    lp_b_ub: Optional[List[float]] = None,
    lp_A_eq: Optional[List[List[float]]] = None,
    lp_b_eq: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
@apply_description("regression_modeler")
def regression_modeler(
    operation: str = "fit",
    x_data: Optional[List[List[float]]] = None,
    y_data: Optional[List[float]] = None,
    model_type: str = "linear",
    degree: int = 2,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    test_size: float = 0.2,
    y_true: Optional[List[float]] = None,
    y_pred: Optional[List[float]] = None,
    models_results: Optional[List[Dict[str, Any]]] = None,
    training_x: Optional[List[List[float]]] = None,
    training_y: Optional[List[float]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
@apply_description("expression_evaluator")
def expression_evaluator(
    expression: str,
    variables: Optional[Dict[str, float]] = None,
    mode: str = "evaluate",
    output_format: str = "decimal",
) -> Dict[str, Any]:
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
@apply_description("create_and_save_chart")
def create_and_save_chart(
    chart_type: str,
    data: Optional[List[float]] = None,
    x_data: Optional[List[float]] = None,
    y_data: Optional[List[float]] = None,
    y_data_series: Optional[List[List[float]]] = None,
    series_labels: Optional[List[str]] = None,
    matrix_data: Optional[List[List[float]]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Statistical Chart",
    xlabel: str = "X Axis",
    ylabel: str = "Y Axis",
    filename: Optional[str] = None,
    format: str = "png",
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    style: str = "whitegrid",
    timestamp: bool = True,
    show_values: bool = False,
    horizontal: bool = False,
    trend_line: bool = False,
    trend_line_color: Optional[str] = None,
    trend_line_equation: Optional[str] = None,
    bins: int = 30,
    annotate: bool = True,
    colormap: str = "viridis",
    color: Optional[str] = None,
    line_width: float = 2.0,
    line_style: str = "-",
    marker: str = "o",
    marker_size: int = 6,
    alpha: float = 0.7,
    grid: bool = True,
) -> Dict[str, Any]:
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
@apply_description("plot_function_curve")
def plot_function_curve(
    function_expression: str,
    variable: str = "x",
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: str = "Function Graph",
    xlabel: str = "X Axis",
    ylabel: str = "Y Axis",
    filename: Optional[str] = None,
    format: str = "png",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    color: str = "blue",
    line_width: float = 2.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    derivative_order: Optional[int] = None,
    show_critical_points: bool = False,
    show_equation: bool = True,
    equation_position: str = "upper right",
    alpha: float = 1.0,
    line_style: str = "-",
    marker: str = "",
    marker_size: int = 6,
) -> Dict[str, Any]:
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
@apply_description("geometry_calculator")
def geometry_calculator(
    shape_type: str,
    operation: str,
    dimensions: Optional[Dict[str, float]] = None,
    points: Optional[List[List[float]]] = None,
    precision: Optional[int] = None,
    unit: str = "default",
) -> Dict[str, Any]:
    try:
        return geometry_calc.geometry_calculator_tool(
            shape_type, operation, dimensions, points, precision, unit
        )
    except Exception as e:
        return {"error": f"Error in geometry calculation: {str(e)}"}


# === Number Theory Tools ===
@mcp.tool()
@apply_description("number_theory_calculator")
def number_theory_calculator(
    operation: str,
    number: Optional[int] = None,
    numbers: Optional[List[int]] = None,
    modulus: Optional[int] = None,
    base: Optional[int] = None,
    exponent: Optional[int] = None,
    limit: Optional[int] = None,
    precision: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        return number_theory_calc.number_theory_tool(
            operation, number, numbers, modulus, base, exponent, limit, precision
        )
    except Exception as e:
        return {"error": f"Error in number theory calculation: {str(e)}"}


# === Signal Processing Tools ===
@mcp.tool()
@apply_description("signal_processing_calculator")
def signal_processing_calculator(
    operation: str,
    signal: Optional[List[float]] = None,
    signal_file: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    frequency: Optional[float] = None,
    filter_type: Optional[str] = None,
    cutoff_freq: Optional[float] = None,
    window_size: Optional[int] = None,
    overlap: Optional[float] = None,
    order: Optional[int] = None,
    duration: Optional[float] = None,
    noise_level: Optional[float] = None,
    signal_type: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
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
@apply_description("financial_calculator")
def financial_calculator(
    operation: str,
    principal: Optional[float] = None,
    rate: Optional[float] = None,
    time: Optional[int] = None,
    cash_flows: Optional[List[float]] = None,
    initial_investment: Optional[float] = None,
    payment: Optional[float] = None,
    periods: Optional[int] = None,
    future_value: Optional[float] = None,
    present_value: Optional[float] = None,
    annual_rate: Optional[float] = None,
    payments_per_year: int = 12,
    risk_free_rate: Optional[float] = None,
    returns: Optional[List[float]] = None,
    prices: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
@apply_description("probability_calculator")
def probability_calculator(
    operation: str,
    distribution: Optional[str] = None,
    parameters: Optional[Dict[str, float]] = None,
    x_value: Optional[float] = None,
    x_values: Optional[List[float]] = None,
    probability: Optional[float] = None,
    n_samples: Optional[int] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    data: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
@apply_description("complex_analysis_suite")
def complex_analysis_suite(
    operation: str,
    complex_number: Optional[Union[str, complex]] = None,
    complex_numbers: Optional[List[Union[str, complex]]] = None,
    function_expression: Optional[str] = None,
    variable: str = "z",
    contour_points: Optional[List[List[float]]] = None,
    singularities: Optional[List[Union[str, complex]]] = None,
    center: Optional[Union[str, complex]] = None,
    radius: Optional[float] = None,
    order: Optional[int] = None,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 500,
    colormap: str = "hsv",
    filename: Optional[str] = None,
    plot_type: str = "domain_coloring",
    series_terms: int = 10,
    branch_cut: Optional[str] = None,
    method: str = "auto",
) -> Dict[str, Any]:
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
@apply_description("graph_theory_suite")
def graph_theory_suite(
    operation: str,
    graph_data: Optional[Dict[str, Any]] = None,
    adjacency_matrix: Optional[List[List[Union[int, float]]]] = None,
    edge_list: Optional[List[List[Union[int, str]]]] = None,
    node_list: Optional[List[Union[int, str]]] = None,
    source_node: Optional[Union[int, str]] = None,
    target_node: Optional[Union[int, str]] = None,
    weight_attribute: str = "weight",
    directed: bool = False,
    algorithm: str = "auto",
    k_value: Optional[int] = None,
    threshold: Optional[float] = None,
    layout: str = "spring",
    filename: Optional[str] = None,
    node_colors: Optional[List[str]] = None,
    edge_colors: Optional[List[str]] = None,
    node_sizes: Optional[List[int]] = None,
    show_labels: bool = True,
    figsize: Tuple[float, float] = (10, 8),
) -> Dict[str, Any]:
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
@apply_description("cleanup_resources")
def cleanup_resources() -> Dict[str, Any]:
    """delete files generated in OUTPUT_PATH(or default temporary directory) and perform basic resource cleanup"""
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
