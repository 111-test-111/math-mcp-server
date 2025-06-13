# -*- coding: utf-8 -*-
"""
复分析计算模块
提供完整丰富的复变函数分析功能
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import cmath
import scipy.integrate as integrate
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

try:
    from .file_utils import generate_unique_filename
except ImportError:
    from math_mcp.file_utils import generate_unique_filename


class ComplexAnalysisCalculator:
    """复分析计算器类，提供完整的复变函数分析功能"""

    def __init__(self):
        """初始化复分析计算器"""
        pass

    def complex_analysis_suite_tool(
        self,
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
        """
        综合复分析计算工具

        Args:
            operation: 操作类型
            complex_number: 单个复数
            complex_numbers: 复数列表
            function_expression: 复函数表达式
            variable: 复变量名
            contour_points: 积分路径点
            singularities: 奇点列表
            center: 展开中心
            radius: 收敛半径
            order: 极点阶数或级数项数
            x_range: 实轴范围
            y_range: 虚轴范围
            resolution: 绘图分辨率
            colormap: 颜色映射
            filename: 保存文件名
            plot_type: 绘图类型
            series_terms: 级数项数
            branch_cut: 分支切割
            method: 计算方法

        Returns:
            复分析计算结果
        """
        try:
            if operation == "convert_form":
                if complex_number is None:
                    return {"error": "转换操作需要提供复数"}
                return self._convert_complex_form(complex_number)

            elif operation == "arithmetic":
                if complex_numbers is None or len(complex_numbers) < 2:
                    return {"error": "复数运算需要至少两个复数"}
                return self._complex_arithmetic(complex_numbers, method)

            elif operation == "function_evaluation":
                if function_expression is None or complex_number is None:
                    return {"error": "函数求值需要函数表达式和复数"}
                return self._evaluate_complex_function(
                    function_expression, complex_number, variable
                )

            elif operation == "residue_calculation":
                if function_expression is None or singularities is None:
                    return {"error": "留数计算需要函数表达式和奇点"}
                return self._calculate_residues(
                    function_expression, singularities, variable, order
                )

            elif operation == "contour_integration":
                if function_expression is None or contour_points is None:
                    return {"error": "围道积分需要函数表达式和积分路径"}
                return self._contour_integration(
                    function_expression, contour_points, variable
                )

            elif operation == "series_expansion":
                if function_expression is None:
                    return {"error": "级数展开需要函数表达式"}
                return self._series_expansion(
                    function_expression, variable, center, series_terms
                )

            elif operation == "analytic_continuation":
                if function_expression is None:
                    return {"error": "解析延拓需要函数表达式"}
                return self._analytic_continuation(
                    function_expression, variable, center, radius, branch_cut
                )

            elif operation == "complex_plot":
                if function_expression is None:
                    return {"error": "复函数绘图需要函数表达式"}
                return self._plot_complex_function(
                    function_expression,
                    variable,
                    x_range,
                    y_range,
                    resolution,
                    plot_type,
                    colormap,
                    filename,
                )

            elif operation == "conformal_mapping":
                if function_expression is None:
                    return {"error": "保形映射需要函数表达式"}
                return self._conformal_mapping(
                    function_expression,
                    variable,
                    x_range,
                    y_range,
                    resolution,
                    filename,
                )

            elif operation == "singularity_analysis":
                if function_expression is None:
                    return {"error": "奇点分析需要函数表达式"}
                return self._analyze_singularities(
                    function_expression, variable, x_range, y_range
                )

            else:
                return {"error": f"不支持的操作类型: {operation}"}

        except Exception as e:
            return {"error": f"复分析计算出错: {str(e)}"}

    def _convert_complex_form(self, z: Union[str, complex]) -> Dict[str, Any]:
        """
        复数形式转换

        Args:
            z: 复数

        Returns:
            各种形式的复数表示
        """
        try:
            # 解析复数
            if isinstance(z, str):
                # 处理字符串形式的复数
                z_parsed = complex(z.replace("i", "j").replace("I", "j"))
            else:
                z_parsed = complex(z)

            # 提取实部和虚部
            real_part = z_parsed.real
            imag_part = z_parsed.imag

            # 计算模长和幅角
            modulus = abs(z_parsed)
            argument = cmath.phase(z_parsed)
            argument_degrees = np.degrees(argument)

            # 指数形式
            exponential_form = f"{modulus:.6f} * exp({argument:.6f}j)"

            # 三角形式
            trigonometric_form = (
                f"{modulus:.6f} * (cos({argument:.6f}) + j*sin({argument:.6f}))"
            )

            # 极坐标形式
            polar_form = f"({modulus:.6f}, {argument:.6f})"
            polar_degrees = f"({modulus:.6f}, {argument_degrees:.6f}°)"

            return {
                "original": str(z),
                "rectangular": {
                    "real": real_part,
                    "imaginary": imag_part,
                    "form": f"{real_part:+.6f}{imag_part:+.6f}j",
                },
                "polar": {
                    "modulus": modulus,
                    "argument_radians": argument,
                    "argument_degrees": argument_degrees,
                    "form_radians": polar_form,
                    "form_degrees": polar_degrees,
                },
                "exponential": {
                    "form": exponential_form,
                    "euler_form": f"{modulus:.6f} * e^({argument:.6f}j)",
                },
                "trigonometric": {"form": trigonometric_form},
                "conjugate": {
                    "value": complex(real_part, -imag_part),
                    "form": f"{real_part:+.6f}{-imag_part:+.6f}j",
                },
            }

        except Exception as e:
            return {"error": f"复数转换出错: {str(e)}"}

    def _complex_arithmetic(
        self, numbers: List[Union[str, complex]], operation: str = "all"
    ) -> Dict[str, Any]:
        """
        复数算术运算

        Args:
            numbers: 复数列表
            operation: 运算类型

        Returns:
            运算结果
        """
        try:
            # 解析复数
            parsed_numbers = []
            for num in numbers:
                if isinstance(num, str):
                    parsed_numbers.append(
                        complex(num.replace("i", "j").replace("I", "j"))
                    )
                else:
                    parsed_numbers.append(complex(num))

            if len(parsed_numbers) < 2:
                return {"error": "需要至少两个复数进行运算"}

            z1, z2 = parsed_numbers[0], parsed_numbers[1]

            results = {"operands": {"z1": str(z1), "z2": str(z2)}}

            if operation in ["all", "add"]:
                results["addition"] = {"result": z1 + z2, "form": str(z1 + z2)}

            if operation in ["all", "subtract"]:
                results["subtraction"] = {"result": z1 - z2, "form": str(z1 - z2)}

            if operation in ["all", "multiply"]:
                results["multiplication"] = {"result": z1 * z2, "form": str(z1 * z2)}

            if operation in ["all", "divide"]:
                if z2 != 0:
                    results["division"] = {"result": z1 / z2, "form": str(z1 / z2)}
                else:
                    results["division"] = {"error": "除零错误"}

            if operation in ["all", "power"]:
                results["power"] = {"z1^z2": z1**z2, "z2^z1": z2**z1}

            # 如果有更多复数，进行链式运算
            if len(parsed_numbers) > 2:
                sum_result = sum(parsed_numbers)
                product_result = parsed_numbers[0]
                for z in parsed_numbers[1:]:
                    product_result *= z

                results["chain_operations"] = {
                    "sum_all": str(sum_result),
                    "product_all": str(product_result),
                }

            return results

        except Exception as e:
            return {"error": f"复数运算出错: {str(e)}"}

    def _evaluate_complex_function(
        self, func_expr: str, z_value: Union[str, complex], variable: str = "z"
    ) -> Dict[str, Any]:
        """
        复函数求值

        Args:
            func_expr: 函数表达式
            z_value: 复数值
            variable: 变量名

        Returns:
            函数值和相关信息
        """
        try:
            # 解析复数
            if isinstance(z_value, str):
                z_parsed = complex(z_value.replace("i", "j").replace("I", "j"))
            else:
                z_parsed = complex(z_value)

            # 使用sympy进行符号计算
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # 计算函数值
            func_value = complex(expr.subs(z_sym, z_parsed))

            # 计算导数
            derivative = sp.diff(expr, z_sym)
            derivative_value = (
                complex(derivative.subs(z_sym, z_parsed)) if derivative else None
            )

            return {
                "function": func_expr,
                "input": {
                    "value": str(z_parsed),
                    "rectangular": f"{z_parsed.real:+.6f}{z_parsed.imag:+.6f}j",
                    "polar": f"({abs(z_parsed):.6f}, {cmath.phase(z_parsed):.6f})",
                },
                "output": {
                    "value": func_value,
                    "rectangular": f"{func_value.real:+.6f}{func_value.imag:+.6f}j",
                    "polar": f"({abs(func_value):.6f}, {cmath.phase(func_value):.6f})",
                    "modulus": abs(func_value),
                    "argument": cmath.phase(func_value),
                },
                "derivative": (
                    {
                        "expression": str(derivative) if derivative else None,
                        "value": str(derivative_value) if derivative_value else None,
                    }
                    if derivative
                    else None
                ),
            }

        except Exception as e:
            return {"error": f"复函数求值出错: {str(e)}"}

    def _calculate_residues(
        self,
        func_expr: str,
        singularities: List[Union[str, complex]],
        variable: str = "z",
        order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        计算留数

        Args:
            func_expr: 函数表达式
            singularities: 奇点列表
            variable: 变量名
            order: 极点阶数

        Returns:
            留数计算结果
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            residues_results = {}
            total_residue = 0

            for i, singularity in enumerate(singularities):
                # 解析奇点
                if isinstance(singularity, str):
                    sing_point = complex(
                        singularity.replace("i", "j").replace("I", "j")
                    )
                else:
                    sing_point = complex(singularity)

                try:
                    # 计算留数
                    residue = sp.residue(expr, z_sym, sing_point)
                    residue_value = complex(residue)
                    total_residue += residue_value

                    # 分析奇点类型
                    # 计算Laurent级数展开
                    try:
                        laurent_series = sp.series(expr, z_sym, sing_point, n=6)
                        singularity_type = self._classify_singularity(
                            laurent_series, sing_point
                        )
                    except:
                        singularity_type = "unknown"

                    residues_results[f"singularity_{i+1}"] = {
                        "point": str(sing_point),
                        "residue": residue_value,
                        "residue_form": str(residue),
                        "type": singularity_type,
                    }

                except Exception as e:
                    residues_results[f"singularity_{i+1}"] = {
                        "point": str(sing_point),
                        "error": f"无法计算留数: {str(e)}",
                    }

            return {
                "function": func_expr,
                "singularities": residues_results,
                "total_residue": total_residue,
                "residue_theorem": f"围道积分值 = 2πi × {total_residue} = {2j * np.pi * total_residue}",
            }

        except Exception as e:
            return {"error": f"留数计算出错: {str(e)}"}

    def _classify_singularity(self, laurent_series, point) -> str:
        """分类奇点类型"""
        try:
            # 简化的奇点分类
            series_str = str(laurent_series)
            if f"1/({point})" in series_str:
                return "simple_pole"
            elif "**(-" in series_str:
                return "pole"
            elif "log" in series_str:
                return "logarithmic_singularity"
            else:
                return "removable_singularity"
        except:
            return "unknown"

    def _contour_integration(
        self, func_expr: str, contour_points: List[List[float]], variable: str = "z"
    ) -> Dict[str, Any]:
        """
        围道积分

        Args:
            func_expr: 函数表达式
            contour_points: 积分路径点
            variable: 变量名

        Returns:
            积分结果
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # 将路径点转换为复数
            contour_complex = [complex(point[0], point[1]) for point in contour_points]

            # 数值积分
            def integrand(t):
                # 参数化路径
                n = len(contour_complex)
                if n < 2:
                    return 0

                # 线性插值路径
                t_scaled = t * (n - 1)
                idx = int(t_scaled)
                if idx >= n - 1:
                    z_t = contour_complex[-1]
                    dz_dt = 0
                else:
                    alpha = t_scaled - idx
                    z_t = (1 - alpha) * contour_complex[idx] + alpha * contour_complex[
                        idx + 1
                    ]
                    dz_dt = contour_complex[idx + 1] - contour_complex[idx]

                # 计算被积函数值
                try:
                    f_val = complex(expr.subs(z_sym, z_t))
                    return f_val * dz_dt * (n - 1)  # 乘以路径长度因子
                except:
                    return 0

            # 数值积分
            real_part, _ = integrate.quad(lambda t: integrand(t).real, 0, 1)
            imag_part, _ = integrate.quad(lambda t: integrand(t).imag, 0, 1)

            integral_result = complex(real_part, imag_part)

            return {
                "function": func_expr,
                "contour": {
                    "points": contour_points,
                    "complex_points": [str(z) for z in contour_complex],
                },
                "integral_result": {
                    "value": integral_result,
                    "real_part": real_part,
                    "imaginary_part": imag_part,
                    "modulus": abs(integral_result),
                    "argument": cmath.phase(integral_result),
                },
            }

        except Exception as e:
            return {"error": f"围道积分计算出错: {str(e)}"}

    def _series_expansion(
        self,
        func_expr: str,
        variable: str = "z",
        center: Optional[Union[str, complex]] = None,
        terms: int = 10,
    ) -> Dict[str, Any]:
        """
        级数展开

        Args:
            func_expr: 函数表达式
            variable: 变量名
            center: 展开中心
            terms: 级数项数

        Returns:
            级数展开结果
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # 解析展开中心
            if center is None:
                center_point = 0
            elif isinstance(center, str):
                center_point = complex(center.replace("i", "j").replace("I", "j"))
            else:
                center_point = complex(center)

            # Taylor级数展开
            try:
                taylor_series = sp.series(expr, z_sym, center_point, n=terms)
                taylor_coefficients = []

                # 提取系数
                for i in range(terms):
                    try:
                        coeff = taylor_series.coeff(z_sym - center_point, i)
                        if coeff is not None:
                            taylor_coefficients.append(complex(coeff))
                        else:
                            taylor_coefficients.append(0)
                    except:
                        taylor_coefficients.append(0)

            except Exception as e:
                taylor_series = f"无法展开: {str(e)}"
                taylor_coefficients = []

            # Laurent级数展开（如果可能）
            try:
                laurent_series = sp.series(expr, z_sym, center_point, n=terms)
                laurent_str = str(laurent_series)
            except:
                laurent_series = "无法进行Laurent展开"
                laurent_str = ""

            # 收敛半径估计
            convergence_radius = self._estimate_convergence_radius(taylor_coefficients)

            return {
                "function": func_expr,
                "expansion_center": str(center_point),
                "taylor_series": {
                    "expression": str(taylor_series),
                    "coefficients": taylor_coefficients,
                    "terms": terms,
                },
                "laurent_series": {"expression": laurent_str},
                "convergence": {
                    "radius": convergence_radius,
                    "domain": (
                        f"|z - {center_point}| < {convergence_radius}"
                        if convergence_radius
                        else "未知"
                    ),
                },
            }

        except Exception as e:
            return {"error": f"级数展开出错: {str(e)}"}

    def _estimate_convergence_radius(
        self, coefficients: List[complex]
    ) -> Optional[float]:
        """估计收敛半径"""
        try:
            if len(coefficients) < 2:
                return None

            # 使用比值判别法
            ratios = []
            for i in range(1, len(coefficients)):
                if coefficients[i] != 0 and coefficients[i - 1] != 0:
                    ratio = abs(coefficients[i - 1] / coefficients[i])
                    ratios.append(ratio)

            if ratios:
                return float(np.mean(ratios))
            return None

        except:
            return None

    def _analytic_continuation(
        self,
        func_expr: str,
        variable: str = "z",
        center: Optional[Union[str, complex]] = None,
        radius: Optional[float] = None,
        branch_cut: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        解析延拓

        Args:
            func_expr: 函数表达式
            variable: 变量名
            center: 延拓中心
            radius: 延拓半径
            branch_cut: 分支切割

        Returns:
            解析延拓结果
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # 解析中心点
            if center is None:
                center_point = 0
            elif isinstance(center, str):
                center_point = complex(center.replace("i", "j").replace("I", "j"))
            else:
                center_point = complex(center)

            # 分析函数的奇点
            singularities = []
            try:
                # 寻找分母为零的点
                denominator = sp.denom(expr)
                if denominator != 1:
                    sing_points = sp.solve(denominator, z_sym)
                    for point in sing_points:
                        try:
                            singularities.append(complex(point))
                        except:
                            singularities.append(str(point))
            except:
                pass

            # 计算延拓域
            if radius is None:
                if singularities:
                    # 找到最近的奇点
                    distances = [
                        abs(complex(s) - center_point)
                        for s in singularities
                        if isinstance(s, (int, float, complex))
                    ]
                    if distances:
                        radius = min(distances) * 0.9  # 略小于最近奇点距离
                    else:
                        radius = 1.0
                else:
                    radius = float("inf")

            # 分支切割分析
            branch_info = {}
            if "log" in func_expr or "sqrt" in func_expr or "**" in func_expr:
                branch_info = {
                    "has_branch_cuts": True,
                    "suggested_cuts": self._analyze_branch_cuts(func_expr),
                    "principal_branch": "使用主分支",
                }
            else:
                branch_info = {"has_branch_cuts": False, "note": "函数在复平面上单值"}

            return {
                "function": func_expr,
                "continuation_center": str(center_point),
                "continuation_radius": radius,
                "domain_of_analyticity": (
                    f"|z - {center_point}| < {radius}"
                    if radius != float("inf")
                    else "整个复平面"
                ),
                "singularities": [str(s) for s in singularities],
                "branch_analysis": branch_info,
                "continuation_method": (
                    "幂级数展开" if radius < float("inf") else "整函数"
                ),
            }

        except Exception as e:
            return {"error": f"解析延拓分析出错: {str(e)}"}

    def _analyze_branch_cuts(self, func_expr: str) -> List[str]:
        """分析分支切割"""
        cuts = []
        if "log" in func_expr:
            cuts.append("负实轴: arg(z) = π")
        if "sqrt" in func_expr:
            cuts.append("负实轴: arg(z) = π")
        if "**" in func_expr and ("1/" in func_expr or "0.5" in func_expr):
            cuts.append("可能的分支切割沿负实轴")
        return cuts

    def _plot_complex_function(
        self,
        func_expr: str,
        variable: str = "z",
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        resolution: int = 500,
        plot_type: str = "domain_coloring",
        colormap: str = "hsv",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        复函数可视化

        Args:
            func_expr: 函数表达式
            variable: 变量名
            x_range: 实轴范围
            y_range: 虚轴范围
            resolution: 分辨率
            plot_type: 绘图类型
            colormap: 颜色映射
            filename: 文件名

        Returns:
            绘图结果
        """
        try:
            # 创建复平面网格
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y

            # 计算函数值
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)
            func_lambdified = sp.lambdify(z_sym, expr, "numpy")

            try:
                W = func_lambdified(Z)
            except:
                # 逐点计算
                W = np.zeros_like(Z, dtype=complex)
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        try:
                            W[i, j] = complex(expr.subs(z_sym, Z[i, j]))
                        except:
                            W[i, j] = np.nan + 1j * np.nan

            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"复函数分析: f(z) = {func_expr}", fontsize=14)

            # 1. 域着色图
            if plot_type in ["domain_coloring", "all"]:
                ax1 = axes[0, 0]

                # 计算颜色
                magnitude = np.abs(W)
                phase = np.angle(W)

                # 归一化
                magnitude_norm = np.log(1 + magnitude) / np.log(
                    1 + np.nanmax(magnitude)
                )
                phase_norm = (phase + np.pi) / (2 * np.pi)

                # HSV颜色空间
                H = phase_norm
                S = np.ones_like(magnitude_norm)
                V = magnitude_norm

                # 转换为RGB
                HSV = np.stack([H, S, V], axis=-1)
                RGB = hsv_to_rgb(HSV)

                ax1.imshow(
                    RGB,
                    extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                    origin="lower",
                    interpolation="bilinear",
                )
                ax1.set_title("域着色图")
                ax1.set_xlabel("Re(z)")
                ax1.set_ylabel("Im(z)")
                ax1.grid(True, alpha=0.3)

            # 2. 模长图
            ax2 = axes[0, 1]
            magnitude_plot = ax2.contourf(X, Y, np.abs(W), levels=20, cmap="viridis")
            ax2.set_title("模长 |f(z)|")
            ax2.set_xlabel("Re(z)")
            ax2.set_ylabel("Im(z)")
            plt.colorbar(magnitude_plot, ax=ax2)

            # 3. 相位图
            ax3 = axes[1, 0]
            phase_plot = ax3.contourf(X, Y, np.angle(W), levels=20, cmap="hsv")
            ax3.set_title("相位 arg(f(z))")
            ax3.set_xlabel("Re(z)")
            ax3.set_ylabel("Im(z)")
            plt.colorbar(phase_plot, ax=ax3)

            # 4. 实部和虚部
            ax4 = axes[1, 1]
            real_plot = ax4.contour(
                X, Y, np.real(W), levels=10, colors="red", alpha=0.7
            )
            imag_plot = ax4.contour(
                X, Y, np.imag(W), levels=10, colors="blue", alpha=0.7
            )
            ax4.set_title("实部(红)和虚部(蓝)")
            ax4.set_xlabel("Re(z)")
            ax4.set_ylabel("Im(z)")
            ax4.legend(["Re(f(z))", "Im(f(z))"])

            plt.tight_layout()

            # 保存图像
            if filename is None:
                filename = "complex_function"

            filepath, _ = generate_unique_filename("complex_function", "png", filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            return {
                "function": func_expr,
                "plot_info": {
                    "type": plot_type,
                    "resolution": f"{resolution}x{resolution}",
                    "domain": f"[{x_range[0]}, {x_range[1]}] + i[{y_range[0]}, {y_range[1]}]",
                },
                "file_path": filepath,
                "analysis": {
                    "max_magnitude": float(np.nanmax(np.abs(W))),
                    "min_magnitude": float(np.nanmin(np.abs(W))),
                    "has_zeros": bool(np.any(np.abs(W) < 1e-10)),
                    "numeric_singularities": bool(np.any(np.isinf(np.abs(W)))),
                    "symbolic_singularities": (
                        sym_sings := self._detect_symbolic_singularities(
                            expr, variable, x_range, y_range
                        )
                    ),
                    "has_singularities": bool(np.any(np.isinf(np.abs(W))))
                    or bool(sym_sings),
                },
            }

        except Exception as e:
            return {"error": f"复函数绘图出错: {str(e)}"}

    def _detect_symbolic_singularities(self, expr, variable, x_range, y_range):
        """使用符号方法检测位于绘图区域内的奇点"""
        try:
            z_sym = sp.Symbol(variable)
            denom = sp.denom(expr)
            if denom == 1:
                return []
            roots = sp.solve(denom, z_sym)
            singular_points = []
            for r in roots:
                try:
                    c = complex(r)
                    if (
                        x_range[0] <= c.real <= x_range[1]
                        and y_range[0] <= c.imag <= y_range[1]
                    ):
                        singular_points.append(str(c))
                except:
                    # 非数值根
                    pass
            return singular_points
        except Exception:
            return []

    def _conformal_mapping(
        self,
        func_expr: str,
        variable: str = "z",
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        resolution: int = 20,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        保形映射可视化

        Args:
            func_expr: 映射函数表达式
            variable: 变量名
            x_range: 实轴范围
            y_range: 虚轴范围
            resolution: 网格分辨率
            filename: 文件名

        Returns:
            保形映射结果
        """
        try:
            # 创建网格
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)

            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)
            func_lambdified = sp.lambdify(z_sym, expr, "numpy")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 原始域
            ax1.set_title("原始域 (z-平面)")
            ax1.set_xlabel("Re(z)")
            ax1.set_ylabel("Im(z)")
            ax1.grid(True)
            ax1.set_aspect("equal")

            # 绘制网格线
            for xi in x[::2]:  # 垂直线
                z_line = xi + 1j * y
                ax1.plot([xi] * len(y), y, "b-", alpha=0.5)

            for yi in y[::2]:  # 水平线
                z_line = x + 1j * yi
                ax1.plot(x, [yi] * len(x), "r-", alpha=0.5)

            # 映射后的域
            ax2.set_title(f"映射域 w = f(z) = {func_expr}")
            ax2.set_xlabel("Re(w)")
            ax2.set_ylabel("Im(w)")
            ax2.grid(True)
            ax2.set_aspect("equal")

            # 映射网格线
            for xi in x[::2]:  # 映射垂直线
                z_line = xi + 1j * y
                try:
                    w_line = func_lambdified(z_line)
                    ax2.plot(np.real(w_line), np.imag(w_line), "b-", alpha=0.7)
                except:
                    pass

            for yi in y[::2]:  # 映射水平线
                z_line = x + 1j * yi
                try:
                    w_line = func_lambdified(z_line)
                    ax2.plot(np.real(w_line), np.imag(w_line), "r-", alpha=0.7)
                except:
                    pass

            plt.tight_layout()

            # 保存图像
            if filename is None:
                filename = "conformal_mapping"

            filepath, _ = generate_unique_filename("conformal_mapping", "png", filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            # 分析保形性质
            jacobian_analysis = self._analyze_jacobian(func_expr, variable)

            return {
                "mapping_function": func_expr,
                "domain": f"[{x_range[0]}, {x_range[1]}] + i[{y_range[0]}, {y_range[1]}]",
                "file_path": filepath,
                "conformal_analysis": jacobian_analysis,
                "grid_resolution": f"{resolution}x{resolution}",
            }

        except Exception as e:
            return {"error": f"保形映射可视化出错: {str(e)}"}

    def _analyze_jacobian(self, func_expr: str, variable: str) -> Dict[str, Any]:
        """分析雅可比矩阵和保形性质"""
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # 计算导数
            derivative = sp.diff(expr, z_sym)

            return {
                "derivative": str(derivative),
                "conformal_condition": "f'(z) ≠ 0",
                "note": "在f'(z) ≠ 0的点处映射是保形的",
            }
        except:
            return {"note": "无法分析保形性质"}

    def _analyze_singularities(
        self,
        func_expr: str,
        variable: str = "z",
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
    ) -> Dict[str, Any]:
        """
        奇点分析

        Args:
            func_expr: 函数表达式
            variable: 变量名
            x_range: 实轴范围
            y_range: 虚轴范围

        Returns:
            奇点分析结果
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            singularities = []

            # 寻找极点（分母为零的点）
            try:
                denominator = sp.denom(expr)
                if denominator != 1:
                    poles = sp.solve(denominator, z_sym)
                    for pole in poles:
                        try:
                            pole_complex = complex(pole)
                            if (
                                x_range[0] <= pole_complex.real <= x_range[1]
                                and y_range[0] <= pole_complex.imag <= y_range[1]
                            ):

                                # 分析极点阶数
                                order = self._find_pole_order(expr, z_sym, pole)
                                singularities.append(
                                    {
                                        "point": str(pole_complex),
                                        "type": "pole",
                                        "order": order,
                                        "location": [
                                            pole_complex.real,
                                            pole_complex.imag,
                                        ],
                                    }
                                )
                        except:
                            singularities.append(
                                {
                                    "point": str(pole),
                                    "type": "pole",
                                    "order": "unknown",
                                    "note": "符号形式的极点",
                                }
                            )
            except:
                pass

            # 寻找支点
            if any(func in func_expr for func in ["sqrt", "log", "**"]):
                branch_points = self._find_branch_points(expr, z_sym)
                singularities.extend(branch_points)

            # 寻找本质奇点
            essential_singularities = self._find_essential_singularities(expr, z_sym)
            singularities.extend(essential_singularities)

            return {
                "function": func_expr,
                "analysis_domain": f"[{x_range[0]}, {x_range[1]}] + i[{y_range[0]}, {y_range[1]}]",
                "singularities": singularities,
                "total_count": len(singularities),
                "classification": self._classify_singularities(singularities),
            }

        except Exception as e:
            return {"error": f"奇点分析出错: {str(e)}"}

    def _find_pole_order(self, expr, variable, pole_point) -> int:
        """确定极点阶数"""
        try:
            # 计算Laurent级数
            series = sp.series(expr, variable, pole_point, n=10)
            series_str = str(series)

            # 寻找最高负幂次
            import re

            negative_powers = re.findall(r"\*\*\(-(\d+)\)", series_str)
            if negative_powers:
                return max(int(power) for power in negative_powers)
            return 1
        except:
            return 1

    def _find_branch_points(self, expr, variable) -> List[Dict[str, Any]]:
        """寻找支点"""
        branch_points = []
        try:
            # 简化的支点检测
            expr_str = str(expr)
            if "log" in expr_str:
                branch_points.append(
                    {
                        "point": "0",
                        "type": "branch_point",
                        "function": "logarithm",
                        "note": "对数函数的支点",
                    }
                )
            if "sqrt" in expr_str:
                branch_points.append(
                    {
                        "point": "0",
                        "type": "branch_point",
                        "function": "square_root",
                        "note": "平方根函数的支点",
                    }
                )
        except:
            pass
        return branch_points

    def _find_essential_singularities(self, expr, variable) -> List[Dict[str, Any]]:
        """寻找本质奇点"""
        essential = []
        try:
            expr_str = str(expr)
            if "exp" in expr_str and ("1/" in expr_str or "**(-" in expr_str):
                essential.append(
                    {
                        "type": "essential_singularity",
                        "note": "可能包含本质奇点，需要进一步分析",
                    }
                )
        except:
            pass
        return essential

    def _classify_singularities(
        self, singularities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """分类奇点统计"""
        classification = {
            "poles": 0,
            "branch_points": 0,
            "essential_singularities": 0,
            "removable_singularities": 0,
        }

        for sing in singularities:
            if sing.get("type") == "pole":
                classification["poles"] += 1
            elif sing.get("type") == "branch_point":
                classification["branch_points"] += 1
            elif sing.get("type") == "essential_singularity":
                classification["essential_singularities"] += 1
            elif sing.get("type") == "removable_singularity":
                classification["removable_singularities"] += 1

        return classification
