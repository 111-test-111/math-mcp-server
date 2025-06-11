# -*- coding: utf-8 -*-
"""
数学计算MCP服务器 - 精简同步版本
用于为LLM提供强大的数学计算工具，采用模块化架构
"""

from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Tuple
import warnings
import os
import base64
from datetime import datetime
import pathlib
import sympy as sp

# 导入各个计算器模块
try:
    # 尝试相对导入（作为包使用时）
    from .matrix_calculator import MatrixCalculator
    from .statistics_calculator import StatisticsCalculator
    from .calculus_calculator import CalculusCalculator
    from .optimization_calculator import OptimizationCalculator
    from .regression_calculator import RegressionCalculator
    from .plotting_calculator import PlottingCalculator
    from .basic_calculator import BasicCalculator
except ImportError:
    # 回退到绝对导入（直接运行时）
    from matrix_calculator import MatrixCalculator
    from statistics_calculator import StatisticsCalculator
    from calculus_calculator import CalculusCalculator
    from optimization_calculator import OptimizationCalculator
    from regression_calculator import RegressionCalculator
    from plotting_calculator import PlottingCalculator
    from basic_calculator import BasicCalculator

warnings.filterwarnings("ignore")

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
def basic_arithmetic(
    operation: str,
    numbers: List[float],
    precision: Optional[int] = None,
    use_decimal: bool = False,
) -> Dict[str, Any]:
    """
    基础算术运算工具

    Args:
        operation: 运算类型 ('add', 'subtract', 'multiply', 'divide', 'power', 'modulo', 'factorial', 'gcd', 'lcm')
        numbers: 数值列表
        precision: 计算精度（小数位数）
        use_decimal: 是否使用高精度小数计算

    Returns:
        计算结果
    """
    try:
        return basic_calc.basic_arithmetic_tool(
            operation, numbers, precision, use_decimal
        )
    except Exception as e:
        return {"error": f"基础算术运算出错: {str(e)}"}


@mcp.tool()
def mathematical_functions(
    function: str,
    value: float,
    base: Optional[float] = None,
    precision: Optional[int] = None,
    angle_unit: str = "radians",
) -> Dict[str, Any]:
    """
    数学函数计算工具

    Args:
        function: 函数类型 ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                           'log', 'log10', 'ln', 'sqrt', 'cbrt', 'exp', 'abs', 'ceil', 'floor', 'round')
        value: 输入值
        base: 对数的底数（可选）
        precision: 结果精度
        angle_unit: 角度单位 ('radians', 'degrees')

    Returns:
        函数计算结果
    """
    try:
        return basic_calc.mathematical_functions_tool(
            function, value, base, precision, angle_unit
        )
    except Exception as e:
        return {"error": f"数学函数计算出错: {str(e)}"}


@mcp.tool()
def number_converter(
    number: str,
    from_base: int = 10,
    to_base: int = 10,
    operation: str = "convert",
    precision: Optional[int] = None,
) -> Dict[str, Any]:
    """
    数值进制转换和格式化工具

    Args:
        number: 输入数值
        from_base: 源进制 (2-36)
        to_base: 目标进制 (2-36)
        operation: 操作类型 ('convert', 'format', 'scientific', 'fraction')
        precision: 精度控制

    Returns:
        转换结果
    """
    try:
        return basic_calc.number_converter_tool(
            number, from_base, to_base, operation, precision
        )
    except Exception as e:
        return {"error": f"数值转换出错: {str(e)}"}


@mcp.tool()
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str,
    unit_type: str,
) -> Dict[str, Any]:
    """
    单位转换工具

    Args:
        value: 输入值
        from_unit: 源单位
        to_unit: 目标单位
        unit_type: 单位类型 ('length', 'weight', 'temperature', 'area', 'volume', 'time', 'speed', 'energy')

    Returns:
        转换结果
    """
    try:
        return basic_calc.unit_converter_tool(value, from_unit, to_unit, unit_type)
    except Exception as e:
        return {"error": f"单位转换出错: {str(e)}"}


@mcp.tool()
def precision_calculator(
    numbers: List[float],
    operation: str,
    precision_digits: int = 10,
    rounding_mode: str = "round_half_up",
) -> Dict[str, Any]:
    """
    高精度计算工具

    Args:
        numbers: 数值列表
        operation: 运算类型 ('add', 'subtract', 'multiply', 'divide', 'power', 'sqrt')
        precision_digits: 精度位数
        rounding_mode: 舍入模式

    Returns:
        高精度计算结果
    """
    try:
        return basic_calc.precision_calculator_tool(
            numbers, operation, precision_digits, rounding_mode
        )
    except Exception as e:
        return {"error": f"高精度计算出错: {str(e)}"}


@mcp.tool()
def number_properties(
    number: float,
    analysis_type: str = "comprehensive",
) -> Dict[str, Any]:
    """
    数值属性分析工具

    Args:
        number: 输入数值
        analysis_type: 分析类型 ('comprehensive', 'prime', 'divisors', 'properties')

    Returns:
        数值属性分析结果
    """
    try:
        return basic_calc.number_properties_tool(number, analysis_type)
    except Exception as e:
        return {"error": f"数值属性分析出错: {str(e)}"}


# === 矩阵计算工具 ===
@mcp.tool()
def matrix_calculator(
    operation: str,
    matrix_a: List[List[float]],
    matrix_b: Optional[List[List[float]]] = None,
    method: Optional[str] = None,
    power: Optional[int] = None,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    综合矩阵计算工具 - 合并所有矩阵相关操作

    Args:
        operation: 运算类型 ('basic', 'decomposition', 'eigenvalues', 'svd', 'properties', 'power', 'exponential', 'solve')
        matrix_a: 第一个矩阵
        matrix_b: 第二个矩阵（某些运算需要）
        method: 方法类型（用于basic操作：'add', 'subtract', 'multiply', 'transpose', 'determinant', 'inverse'；
                          用于decomposition：'qr', 'lu'；
                          用于properties：'rank', 'trace', 'condition_number', 'norm'）
        power: 矩阵幂次（用于power操作）
        property_type: 属性类型（向后兼容）

    Returns:
        计算结果
    """
    try:
        return matrix_calc.matrix_calculator_tool(
            operation, matrix_a, matrix_b, method, power, property_type
        )
    except Exception as e:
        return {"error": f"矩阵计算出错: {str(e)}"}


# === 统计分析工具 ===
@mcp.tool()
def statistics_analyzer(
    data1: List[float],
    analysis_type: str,
    data2: Optional[List[float]] = None,
    test_type: Optional[str] = None,
    confidence: float = 0.95,
    distribution_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    综合统计分析工具 - 合并所有统计相关操作

    Args:
        data1: 第一组数据
        analysis_type: 分析类型 ('descriptive', 'tests', 'distribution', 'confidence_interval')
        data2: 第二组数据（某些分析需要）
        test_type: 检验类型（'normality', 'hypothesis', 'correlation'）或分布分析类型（'fitting', 'percentiles', 'outliers'）
        confidence: 置信水平
        distribution_type: 分布类型（用于分布拟合）

    Returns:
        统计分析结果
    """
    try:
        return stats_calc.statistics_analyzer_tool(
            data1, analysis_type, data2, test_type, confidence, distribution_type
        )
    except Exception as e:
        return {"error": f"统计分析出错: {str(e)}"}


# === 微积分计算工具 ===
@mcp.tool()
def calculus_engine(
    expression: str,
    operation: str,
    variable: str = "x",
    variables: Optional[List[str]] = None,
    limits: Optional[List[float]] = None,
    point: Optional[float] = None,
    points: Optional[List[float]] = None,
    order: int = 2,
    method: str = "quad",
    mode: str = "symbolic",
) -> Dict[str, Any]:
    """
    综合微积分计算工具

    Args:
        expression: 数学表达式字符串
        operation: 运算类型 ('derivative', 'integral', 'limit', 'series', 'critical_points', 'partial', 'gradient', 'taylor', 'arc_length')
        variable: 主变量名
        variables: 多变量列表（用于偏导数、梯度等）
        limits: 积分限或其他范围
        point: 计算点
        points: 多个计算点
        order: 导数阶数或泰勒级数项数
        method: 计算方法
        mode: 计算模式 ('symbolic', 'numerical')

    Returns:
        微积分计算结果
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
        return {"error": f"微积分计算出错: {str(e)}"}


# === 优化计算工具 ===
@mcp.tool()
def optimization_suite(
    objective_function: str,
    variables: List[str],
    operation: str = "minimize",
    method: str = "symbolic",
    initial_guess: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    constraints: Optional[List[str]] = None,
    equation: Optional[str] = None,
    root_method: str = "fsolve",
    lp_c: Optional[List[float]] = None,
    lp_A_ub: Optional[List[List[float]]] = None,
    lp_b_ub: Optional[List[float]] = None,
    lp_A_eq: Optional[List[List[float]]] = None,
    lp_b_eq: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    综合优化计算工具

    Args:
        objective_function: 目标函数表达式
        variables: 变量列表
        operation: 操作类型 ('minimize', 'maximize', 'root_finding', 'linear_programming', 'least_squares', 'constrained', 'global')
        method: 计算方法
        initial_guess: 初始猜测值
        bounds: 变量边界
        constraints: 约束条件
        equation: 方程（用于求根）
        root_method: 求根方法
        lp_c: 线性规划目标函数系数
        lp_A_ub: 线性规划不等式约束矩阵
        lp_b_ub: 线性规划不等式约束向量
        lp_A_eq: 线性规划等式约束矩阵
        lp_b_eq: 线性规划等式约束向量

    Returns:
        优化计算结果
    """
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
def regression_modeler(
    x_data: List[List[float]],
    y_data: List[float],
    model_type: str = "linear",
    degree: int = 2,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    综合回归建模工具

    Args:
        x_data: 自变量数据矩阵
        y_data: 因变量数据
        model_type: 模型类型 ('linear', 'polynomial', 'ridge', 'lasso', 'elastic_net', 'logistic')
        degree: 多项式次数（用于多项式回归）
        alpha: 正则化参数（用于正则化回归）
        l1_ratio: L1正则化比例（用于ElasticNet）
        cv_folds: 交叉验证折数
        test_size: 测试集比例

    Returns:
        回归建模结果
    """
    try:
        return regression_calc.regression_modeler_tool(
            x_data, y_data, model_type, degree, alpha, l1_ratio, cv_folds, test_size
        )
    except Exception as e:
        return {"error": f"回归建模出错: {str(e)}"}


# === 表达式求值工具 ===
@mcp.tool()
def expression_evaluator(
    expression: str,
    variables: Optional[Dict[str, float]] = None,
    mode: str = "evaluate",
    output_format: str = "decimal",
) -> Dict[str, Any]:
    """
    数学表达式求值和简化工具

    Args:
        expression: 数学表达式字符串
        variables: 变量值字典
        mode: 计算模式 ('evaluate', 'simplify', 'expand', 'factor')
        output_format: 输出格式 ('decimal', 'fraction', 'latex')

    Returns:
        表达式计算结果
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
        return {"error": f"表达式计算出错: {str(e)}"}


# === 绘图工具 ===
@mcp.tool()
def create_and_save_chart(
    chart_type: str,
    data: Optional[List[float]] = None,
    x_data: Optional[List[float]] = None,
    y_data: Optional[List[float]] = None,
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
    # 图表特定参数
    show_values: bool = False,
    horizontal: bool = False,
    trend_line: bool = False,
    trend_line_color: Optional[str] = None,
    trend_line_equation: Optional[str] = None,
    bins: int = 30,
    annotate: bool = True,
    colormap: str = "viridis",
    # 线条和颜色参数
    color: Optional[str] = None,
    line_width: float = 2.0,
    line_style: str = "-",
    marker: str = "o",
    marker_size: int = 6,
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """
    创建并保存统计图表（推荐使用）

    Args:
        chart_type: 图表类型 ('bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap')
        data: 单组数据（用于柱状图、饼图、直方图、箱线图）
        x_data: X轴数据（用于线图、散点图）
        y_data: Y轴数据（用于线图、散点图）
        matrix_data: 矩阵数据（用于热力图、多组箱线图）
        labels: 数据标签
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        filename: 保存文件名
        colors: 颜色列表
        figsize: 图片大小 (width, height)
        dpi: 图片分辨率
        style: 图表样式
        timestamp: 是否在文件名中添加时间戳
        format: 图片格式
        show_values: 是否显示数值（柱状图）
        horizontal: 是否为水平图表（柱状图）
        trend_line: 是否显示趋势线（散点图）
        trend_line_color: 趋势线颜色
        trend_line_equation: 趋势线方程
        bins: 直方图分箱数量
        annotate: 是否显示标注（热力图）
        colormap: 颜色映射（热力图）
        color: 线条/点的颜色
        line_width: 线条宽度
        line_style: 线条样式
        marker: 标记样式
        marker_size: 标记大小
        alpha: 透明度

    Returns:
        保存结果
    """
    try:
        # 生成文件名
        if filename is None:
            filename = f"{chart_type}_chart"

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp_str}"

        # 添加文件扩展名
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        # 确保输出目录存在
        output_path = os.getenv("OUTPUT_PATH", "plots")
        output_dir = pathlib.Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        filepath = output_dir / filename

        # 构建参数字典
        plot_kwargs = {
            "show_values": show_values,
            "horizontal": horizontal,
            "trend_line": trend_line,
            "trend_line_color": trend_line_color,
            "trend_line_equation": trend_line_equation,
            "bins": bins,
            "annotate": annotate,
            "colormap": colormap,
            "color": color,
            "line_width": line_width,
            "line_style": line_style,
            "marker": marker,
            "marker_size": marker_size,
            "alpha": alpha,
        }

        # 调用绘图功能获取base64图片
        plot_result = plotting_calc.statistical_plotter_tool(
            chart_type,
            data,
            x_data,
            y_data,
            matrix_data,
            labels,
            title,
            xlabel,
            ylabel,
            colors,
            figsize,
            dpi,
            style,
            **plot_kwargs,
        )

        if "error" in plot_result:
            return plot_result

        # 保存图片到文件
        if "image_base64" in plot_result:
            image_data = base64.b64decode(plot_result["image_base64"])
            with open(filepath, "wb") as f:
                f.write(image_data)

            return {
                "success": True,
                "chart_type": chart_type,
                "filename": str(filepath),
                "absolute_path": str(filepath.absolute()),
                "message": f"图表已保存到: {filepath}",
            }
        else:
            return {"error": "绘图结果中没有图片数据"}

    except Exception as e:
        return {"error": f"图表创建出错: {str(e)}"}


@mcp.tool()
def plot_function_curve(
    function_expression: str,
    variable: str = "x",
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "f(x)",
    filename: Optional[str] = None,
    parameters: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    color: str = "blue",
    line_width: float = 2.0,
    line_style: str = "-",
    marker: str = "o",
    marker_size: int = 6,
    alpha: float = 0.7,
    grid: bool = True,
    timestamp: bool = True,
    format: str = "png",
) -> Dict[str, Any]:
    """
    绘制函数曲线图

    Args:
        function_expression: 函数表达式
        variable: 变量名
        x_range: X轴范围
        num_points: 采样点数
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        filename: 保存文件名
        parameters: 函数参数字典
        figsize: 图片大小
        dpi: 图片分辨率
        color: 线条颜色
        line_width: 线条宽度
        line_style: 线条样式
        marker: 标记样式
        marker_size: 标记大小
        alpha: 透明度
        grid: 是否显示网格
        timestamp: 是否添加时间戳
        format: 图片格式

    Returns:
        绘图结果
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 设置中文字体支持
        font_path = os.getenv("FONT_PATH")
        if font_path and os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams["font.family"] = font_prop.get_name()
                plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
            except Exception as e:
                # 回退到默认中文字体
                plt.rcParams["font.sans-serif"] = [
                    "SimHei",
                    "Microsoft YaHei",
                    "DejaVu Sans",
                    "Arial Unicode MS",
                    "sans-serif",
                ]
        else:
            plt.rcParams["font.sans-serif"] = [
                "SimHei",
                "Microsoft YaHei",
                "DejaVu Sans",
                "Arial Unicode MS",
                "sans-serif",
            ]
        plt.rcParams["axes.unicode_minus"] = False

        # 生成x数据
        x_vals = np.linspace(x_range[0], x_range[1], num_points)

        # 解析函数表达式
        var = sp.Symbol(variable)
        if parameters:
            # 替换参数
            param_symbols = {param: sp.Symbol(param) for param in parameters.keys()}
            expr = sp.sympify(function_expression)
            expr = expr.subs(parameters)
        else:
            expr = sp.sympify(function_expression)

        # 转换为数值函数
        func = sp.lambdify(var, expr, "numpy")

        # 计算y值
        y_vals = func(x_vals)

        # 创建图表
        if figsize is None:
            figsize = (10, 6)

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=alpha,
            label=f"f({variable}) = {function_expression}",
        )

        if title is None:
            title = f"函数图像: f({variable}) = {function_expression}"

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        if grid:
            plt.grid(True, alpha=0.3)

        plt.legend()

        # 生成文件名
        if filename is None:
            filename = "function_curve"

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp_str}"

        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        # 保存图片
        output_path = os.getenv("OUTPUT_PATH", "plots")
        output_dir = pathlib.Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        filepath = output_dir / filename

        plt.savefig(filepath, format=format, bbox_inches="tight", facecolor="white")
        plt.close()

        return {
            "success": True,
            "function": function_expression,
            "variable": variable,
            "x_range": x_range,
            "filename": str(filepath),
            "absolute_path": str(filepath.absolute()),
            "message": f"函数图像已保存到: {filepath}",
        }

    except Exception as e:
        return {"error": f"函数绘图出错: {str(e)}"}


@mcp.tool()
def statistical_plotter(
    chart_type: str,
    data: Optional[List[float]] = None,
    x_data: Optional[List[float]] = None,
    y_data: Optional[List[float]] = None,
    matrix_data: Optional[List[List[float]]] = None,
    labels: Optional[List[str]] = None,
    title: str = "统计图表",
    xlabel: str = "X轴",
    ylabel: str = "Y轴",
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    style: str = "whitegrid",
    # 图表特定参数
    show_values: bool = False,
    horizontal: bool = False,
    trend_line: bool = False,
    trend_line_color: Optional[str] = None,
    trend_line_equation: Optional[str] = None,
    bins: int = 30,
    annotate: bool = True,
    colormap: str = "viridis",
    # 线条和颜色参数
    color: Optional[str] = None,
    line_width: float = 2.0,
    line_style: str = "-",
    marker: str = "o",
    marker_size: int = 6,
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """
    综合统计绘图工具 - 返回base64编码图片

    ⚠️ 注意：建议使用 create_and_save_chart 工具，它直接保存图片到本地，
    不返回base64数据，更高效且节省大量tokens。
    此工具返回base64数据，仅在需要获取图片数据时使用。

    Args:
        chart_type: 图表类型 ('bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap')
        data: 单组数据
        x_data: X轴数据
        y_data: Y轴数据
        matrix_data: 矩阵数据
        labels: 数据标签
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        colors: 颜色列表
        figsize: 图片大小
        dpi: 图片分辨率
        style: 图表样式
        show_values: 是否显示数值（柱状图）
        horizontal: 是否为水平图表（柱状图）
        trend_line: 是否显示趋势线（散点图）
        trend_line_color: 趋势线颜色
        trend_line_equation: 趋势线方程
        bins: 直方图分箱数量
        annotate: 是否显示标注（热力图）
        colormap: 颜色映射（热力图）
        color: 线条/点的颜色
        line_width: 线条宽度
        line_style: 线条样式
        marker: 标记样式
        marker_size: 标记大小
        alpha: 透明度

    Returns:
        包含base64编码图片的结果字典
    """
    try:
        # 构建参数字典
        plot_kwargs = {
            "show_values": show_values,
            "horizontal": horizontal,
            "trend_line": trend_line,
            "trend_line_color": trend_line_color,
            "trend_line_equation": trend_line_equation,
            "bins": bins,
            "annotate": annotate,
            "colormap": colormap,
            "color": color,
            "line_width": line_width,
            "line_style": line_style,
            "marker": marker,
            "marker_size": marker_size,
            "alpha": alpha,
        }

        return plotting_calc.statistical_plotter_tool(
            chart_type,
            data,
            x_data,
            y_data,
            matrix_data,
            labels,
            title,
            xlabel,
            ylabel,
            colors,
            figsize,
            dpi,
            style,
            **plot_kwargs,
        )
    except Exception as e:
        return {"error": f"统计绘图出错: {str(e)}"}


# === 系统工具 ===
@mcp.tool()
def cleanup_resources() -> Dict[str, Any]:
    """
    清理系统资源，释放内存

    Returns:
        清理结果
    """
    try:
        import matplotlib.pyplot as plt
        import gc

        # 关闭所有matplotlib图形
        plt.close("all")

        # 强制垃圾回收
        collected = gc.collect()

        return {
            "success": True,
            "message": "资源清理完成",
            "collected_objects": collected,
        }
    except Exception as e:
        return {"error": f"资源清理出错: {str(e)}"}


# MCP服务器启动函数
def main():
    """启动MCP服务器"""
    try:
        # 直接运行FastMCP服务器
        mcp.run()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
    except Exception as e:
        print(f"服务器运行出错: {e}")
    finally:
        try:
            # 清理matplotlib资源
            import matplotlib.pyplot as plt

            plt.close("all")

            # 强制垃圾回收
            import gc

            gc.collect()

            print("服务器已安全关闭")
        except Exception as e:
            print(f"关闭过程中出错: {e}")


if __name__ == "__main__":
    main()
