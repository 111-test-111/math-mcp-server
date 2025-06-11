# -*- coding: utf-8 -*-
"""
数学计算MCP服务器 - 动态描述加载版本
用于为LLM提供强大的数学计算工具，采用模块化架构
工具描述信息从外部配置文件动态加载
"""

from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Tuple
import os
import base64
from datetime import datetime
import pathlib
import sympy as sp


# 导入各个计算器模块
try:
    from .description_loader import apply_description
    from .matrix_calculator import MatrixCalculator
    from .statistics_calculator import StatisticsCalculator
    from .calculus_calculator import CalculusCalculator
    from .optimization_calculator import OptimizationCalculator
    from .regression_calculator import RegressionCalculator
    from .plotting_calculator import PlottingCalculator
    from .basic_calculator import BasicCalculator
except ImportError:
    from math_mcp.description_loader import apply_description
    from math_mcp.matrix_calculator import MatrixCalculator
    from math_mcp.statistics_calculator import StatisticsCalculator
    from math_mcp.calculus_calculator import CalculusCalculator
    from math_mcp.optimization_calculator import OptimizationCalculator
    from math_mcp.regression_calculator import RegressionCalculator
    from math_mcp.plotting_calculator import PlottingCalculator
    from math_mcp.basic_calculator import BasicCalculator

# 创建FastMCP应用
mcp = FastMCP("math-calculator")

# 初始化各个计算器
matrix_calc = MatrixCalculator()
stats_calc = StatisticsCalculator()
calculus_calc = CalculusCalculator()
optimization_calc = OptimizationCalculator()
regression_calc = RegressionCalculator()
plotting_calc = PlottingCalculator()
basic_calc = BasicCalculator()


# === 基础数值计算工具 ===
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
        return {"error": f"基础算术运算出错: {str(e)}"}


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
        return {"error": f"数学函数计算出错: {str(e)}"}


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
        return {"error": f"数值转换出错: {str(e)}"}


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
        return {"error": f"单位转换出错: {str(e)}"}


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
        return {"error": f"高精度计算出错: {str(e)}"}


@mcp.tool()
@apply_description("number_properties")
def number_properties(
    number: float,
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    try:
        return basic_calc.number_properties_tool(number, analysis_type)
    except Exception as e:
        return {"error": f"数值属性分析出错: {str(e)}"}


# === 矩阵计算工具 ===
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
        return {"error": f"矩阵计算出错: {str(e)}"}


# === 统计分析工具 ===
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
        return {"error": f"统计分析出错: {str(e)}"}


# === 微积分计算工具 ===
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
        return {"error": f"微积分计算出错: {str(e)}"}


# === 优化计算工具 ===
@mcp.tool()
@apply_description("optimization_suite")
def optimization_suite(
    objective_function: str,
    variables: List[str],
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
            objective_function,
            variables,
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
        return {"error": f"优化计算出错: {str(e)}"}


# === 回归建模工具 ===
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
        )
    except Exception as e:
        return {"error": f"回归建模出错: {str(e)}"}


# === 表达式求值工具 ===
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
        return {"error": f"表达式计算出错: {str(e)}"}


# === 绘图工具 ===
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
    title: str = "统计图表",
    xlabel: str = "X轴",
    ylabel: str = "Y轴",
    filename: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    style: str = "whitegrid",
    timestamp: bool = True,
    format: str = "png",
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
) -> Dict[str, Any]:
    try:
        chart_result = plotting_calc.statistical_plotter_tool(
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
        )

        if chart_result.get("error"):
            return chart_result

        if chart_result.get("image_base64"):
            output_path = os.getenv("OUTPUT_PATH", "plots")
            output_dir = pathlib.Path(output_path)
            output_dir.mkdir(exist_ok=True, parents=True)

            if filename is None:
                timestamp_str = (
                    datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
                )
                filename = (
                    f"{chart_type}_chart_{timestamp_str}"
                    if timestamp_str
                    else f"{chart_type}_chart"
                )

            base_filename = filename.replace(f".{format}", "")
            full_filename = f"{base_filename}.{format}"
            full_path = output_dir / full_filename

            image_data = base64.b64decode(chart_result["image_base64"])
            with open(full_path, "wb") as f:
                f.write(image_data)

            return {
                "success": True,
                "chart_type": chart_type,
                "saved_path": str(full_path.absolute()),
                "filename": full_filename,
                "title": title,
                "message": f"图表已成功保存到: {full_path}",
            }
        else:
            return {"error": "绘图过程中未生成图像数据"}

    except Exception as e:
        return {"error": f"图表创建和保存出错: {str(e)}"}


@mcp.tool()
@apply_description("plot_function_curve")
def plot_function_curve(
    function_expression: str,
    variable: str = "x",
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: str = "函数图像",
    xlabel: str = "X轴",
    ylabel: str = "Y轴",
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
        return {"error": f"函数绘图出错: {str(e)}"}


@mcp.tool()
@apply_description("cleanup_resources")
def cleanup_resources() -> Dict[str, Any]:
    try:
        import gc
        import matplotlib.pyplot as plt

        plt.close("all")
        gc.collect()

        return {
            "success": True,
            "message": "资源清理完成",
            "actions": ["关闭matplotlib图形", "执行垃圾回收"],
        }
    except Exception as e:
        return {"error": f"资源清理出错: {str(e)}"}


# MCP服务器启动函数
def main():
    """启动MCP服务器"""
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
    except Exception as e:
        print(f"服务器运行出错: {e}")
    finally:
        try:
            import matplotlib.pyplot as plt
            import gc

            plt.close("all")
            gc.collect()
            print("服务器已安全关闭")
        except Exception as e:
            print(f"关闭过程中出错: {e}")


if __name__ == "__main__":
    main()
