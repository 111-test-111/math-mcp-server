# -*- coding: utf-8 -*-
"""
微积分计算模块
提供完整丰富的微积分运算功能
"""

import sympy as sp
import numpy as np
import scipy.integrate as integrate
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class CalculusCalculator:
    """微积分计算器类，提供完整的微积分运算功能"""

    def __init__(self):
        """初始化微积分计算器"""
        pass

    def calculus_engine_tool(
        self,
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
            if operation == "derivative":
                if mode == "numerical" and point is not None:
                    return self.numerical_derivative(expression, point, variable)
                else:
                    if order > 1:
                        return self.higher_order_derivatives(
                            expression, variable, order, point
                        )
                    else:
                        return self.symbolic_operations(
                            expression, "derivative", variable, limits, point
                        )
            elif operation == "integral":
                if mode == "numerical" and limits:
                    return self.numerical_integration(
                        expression, limits, variable, method
                    )
                else:
                    return self.symbolic_operations(
                        expression, "integral", variable, limits, point
                    )
            elif operation == "limit":
                return self.symbolic_operations(
                    expression, "limit", variable, limits, point
                )
            elif operation == "series":
                return self.symbolic_operations(
                    expression, "series", variable, limits, point
                )
            elif operation == "critical_points":
                return self.critical_points(expression, variable)
            elif operation == "partial" and variables:
                return self.partial_derivatives(expression, variables)
            elif operation == "gradient" and variables:
                return self.gradient(expression, variables, points)
            elif operation == "taylor":
                return self.taylor_series(expression, variable, point or 0, order)
            elif operation == "arc_length" and limits:
                return self.arc_length(expression, variable, limits)
            else:
                return {"error": f"不支持的运算类型或缺少必要参数: {operation}"}
        except Exception as e:
            return {"error": f"微积分计算出错: {str(e)}"}

    def symbolic_operations(
        self,
        expression: str,
        operation: str,
        variable: str = "x",
        limits: Optional[List[float]] = None,
        point: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        符号微积分运算

        Args:
            expression: 数学表达式字符串
            operation: 运算类型 ('derivative', 'integral', 'limit', 'series')
            variable: 变量名
            limits: 积分限或级数展开点
            point: 求导点或极限点

        Returns:
            运算结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            if operation == "derivative":
                if point is not None:
                    # 在指定点的导数值
                    derivative = sp.diff(expr, var)
                    result_value = float(derivative.subs(var, point))
                    return {
                        "derivative": str(derivative),
                        "value_at_point": result_value,
                        "point": point,
                    }
                else:
                    # 符号导数
                    derivative = sp.diff(expr, var)
                    return {"derivative": str(derivative)}

            elif operation == "integral":
                if limits is not None and len(limits) == 2:
                    # 定积分
                    definite_integral = sp.integrate(expr, (var, limits[0], limits[1]))
                    result_value = float(definite_integral.evalf())
                    return {
                        "definite_integral": str(definite_integral),
                        "value": result_value,
                        "limits": limits,
                    }
                else:
                    # 不定积分
                    indefinite_integral = sp.integrate(expr, var)
                    return {"indefinite_integral": str(indefinite_integral)}

            elif operation == "limit":
                if point is None:
                    return {"error": "极限运算需要指定点"}
                limit_result = sp.limit(expr, var, point)
                return {"limit": str(limit_result), "point": point}

            elif operation == "series":
                if point is None:
                    point = 0
                n_terms = limits[0] if limits else 6
                series_result = sp.series(expr, var, point, n=int(n_terms))
                return {
                    "series": str(series_result),
                    "point": point,
                    "terms": int(n_terms),
                }

            else:
                return {"error": f"不支持的运算类型: {operation}"}

        except Exception as e:
            return {"error": f"符号计算出错: {str(e)}"}

    def numerical_derivative(
        self, function_expr: str, point: float, variable: str = "x", h: float = 1e-5
    ) -> Dict[str, Any]:
        """
        数值求导

        Args:
            function_expr: 函数表达式
            point: 求导点
            variable: 变量名
            h: 步长

        Returns:
            数值导数结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(function_expr)
            func = sp.lambdify(var, expr, "numpy")

            # 使用中心差分法
            derivative = (func(point + h) - func(point - h)) / (2 * h)

            return {
                "numerical_derivative": float(derivative),
                "point": point,
                "step_size": h,
                "method": "central_difference",
            }
        except Exception as e:
            return {"error": f"数值求导出错: {str(e)}"}

    def numerical_integration(
        self,
        function_expr: str,
        limits: List[float],
        variable: str = "x",
        method: str = "quad",
    ) -> Dict[str, Any]:
        """
        数值积分

        Args:
            function_expr: 函数表达式
            limits: 积分区间 [a, b]
            variable: 变量名
            method: 积分方法 ('quad', 'simpson', 'trapz')

        Returns:
            数值积分结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(function_expr)
            func = sp.lambdify(var, expr, "numpy")

            a, b = limits[0], limits[1]

            if method == "quad":
                result, error = integrate.quad(func, a, b)
                return {
                    "integral_value": float(result),
                    "error_estimate": float(error),
                    "method": "adaptive_quadrature",
                    "limits": limits,
                }

            elif method == "simpson":
                # 使用辛普森法则
                n = 1000  # 分割数
                x = np.linspace(a, b, n + 1)
                y = func(x)
                result = integrate.simpson(y, x)
                return {
                    "integral_value": float(result),
                    "method": "simpson_rule",
                    "divisions": n,
                    "limits": limits,
                }

            elif method == "trapz":
                # 使用梯形法则
                n = 1000  # 分割数
                x = np.linspace(a, b, n + 1)
                y = func(x)
                result = integrate.trapz(y, x)
                return {
                    "integral_value": float(result),
                    "method": "trapezoidal_rule",
                    "divisions": n,
                    "limits": limits,
                }

            else:
                return {"error": f"不支持的积分方法: {method}"}

        except Exception as e:
            return {"error": f"数值积分出错: {str(e)}"}

    def higher_order_derivatives(
        self, expression: str, variable: str = "x", order: int = 2
    ) -> Dict[str, Any]:
        """
        高阶导数

        Args:
            expression: 数学表达式
            variable: 变量名
            order: 导数阶数

        Returns:
            高阶导数结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            derivatives = {}
            for i in range(1, order + 1):
                derivative = sp.diff(expr, var, i)
                derivatives[f"order_{i}"] = str(derivative)

            return {
                "expression": expression,
                "variable": variable,
                "derivatives": derivatives,
            }
        except Exception as e:
            return {"error": f"高阶导数计算出错: {str(e)}"}

    def partial_derivatives(
        self, expression: str, variables: List[str]
    ) -> Dict[str, Any]:
        """
        偏导数

        Args:
            expression: 多变量表达式
            variables: 变量列表

        Returns:
            偏导数结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            expr = sp.sympify(expression)

            partial_derivatives = {}
            for var_name, var_symbol in zip(variables, vars_symbols):
                partial_deriv = sp.diff(expr, var_symbol)
                partial_derivatives[f"d/d{var_name}"] = str(partial_deriv)

            return {
                "expression": expression,
                "variables": variables,
                "partial_derivatives": partial_derivatives,
            }
        except Exception as e:
            return {"error": f"偏导数计算出错: {str(e)}"}

    def gradient(
        self, expression: str, variables: List[str], point: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        梯度计算

        Args:
            expression: 多变量函数表达式
            variables: 变量列表
            point: 计算梯度的点（可选）

        Returns:
            梯度结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            expr = sp.sympify(expression)

            # 计算梯度（偏导数向量）
            gradient_components = []
            for var_symbol in vars_symbols:
                partial_deriv = sp.diff(expr, var_symbol)
                gradient_components.append(partial_deriv)

            gradient_expr = [str(comp) for comp in gradient_components]

            result = {
                "expression": expression,
                "variables": variables,
                "gradient": gradient_expr,
            }

            # 如果提供了点，计算该点的梯度值
            if point is not None and len(point) == len(variables):
                substitutions = dict(zip(vars_symbols, point))
                gradient_values = []
                for comp in gradient_components:
                    value = float(comp.subs(substitutions))
                    gradient_values.append(value)

                result["gradient_at_point"] = gradient_values
                result["point"] = point

            return result
        except Exception as e:
            return {"error": f"梯度计算出错: {str(e)}"}

    def taylor_series(
        self, expression: str, variable: str = "x", point: float = 0, order: int = 5
    ) -> Dict[str, Any]:
        """
        泰勒级数展开

        Args:
            expression: 函数表达式
            variable: 变量名
            point: 展开点
            order: 展开阶数

        Returns:
            泰勒级数结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            taylor_expansion = sp.series(expr, var, point, n=order + 1).removeO()

            return {
                "expression": expression,
                "variable": variable,
                "expansion_point": point,
                "order": order,
                "taylor_series": str(taylor_expansion),
            }
        except Exception as e:
            return {"error": f"泰勒级数展开出错: {str(e)}"}

    def critical_points(self, expression: str, variable: str = "x") -> Dict[str, Any]:
        """
        求临界点

        Args:
            expression: 函数表达式
            variable: 变量名

        Returns:
            临界点结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            # 一阶导数
            first_derivative = sp.diff(expr, var)

            # 求解一阶导数为0的点
            critical_points = sp.solve(first_derivative, var)

            # 二阶导数测试
            second_derivative = sp.diff(expr, var, 2)

            point_analysis = []
            for point in critical_points:
                try:
                    second_deriv_value = float(second_derivative.subs(var, point))
                    if second_deriv_value > 0:
                        point_type = "local_minimum"
                    elif second_deriv_value < 0:
                        point_type = "local_maximum"
                    else:
                        point_type = "inflection_point_candidate"

                    function_value = float(expr.subs(var, point))

                    point_analysis.append(
                        {
                            "point": float(point),
                            "function_value": function_value,
                            "type": point_type,
                            "second_derivative": second_deriv_value,
                        }
                    )
                except:
                    point_analysis.append(
                        {
                            "point": str(point),
                            "function_value": "undefined",
                            "type": "complex_or_undefined",
                            "second_derivative": "undefined",
                        }
                    )

            return {
                "expression": expression,
                "first_derivative": str(first_derivative),
                "second_derivative": str(second_derivative),
                "critical_points": point_analysis,
            }
        except Exception as e:
            return {"error": f"临界点分析出错: {str(e)}"}

    def arc_length(
        self, expression: str, variable: str = "x", limits: List[float] = None
    ) -> Dict[str, Any]:
        """
        弧长计算

        Args:
            expression: 函数表达式 y = f(x)
            variable: 变量名
            limits: 积分区间 [a, b]

        Returns:
            弧长结果
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            # 计算导数
            derivative = sp.diff(expr, var)

            # 弧长积分表达式: sqrt(1 + (dy/dx)^2)
            arc_length_integrand = sp.sqrt(1 + derivative**2)

            if limits is not None and len(limits) == 2:
                # 计算定积分（弧长）
                try:
                    arc_length_value = sp.integrate(
                        arc_length_integrand, (var, limits[0], limits[1])
                    )
                    numerical_value = float(arc_length_value.evalf())

                    return {
                        "expression": expression,
                        "derivative": str(derivative),
                        "arc_length_integrand": str(arc_length_integrand),
                        "arc_length": numerical_value,
                        "limits": limits,
                    }
                except:
                    return {
                        "expression": expression,
                        "derivative": str(derivative),
                        "arc_length_integrand": str(arc_length_integrand),
                        "note": "无法计算精确值，需要数值积分",
                    }
            else:
                return {
                    "expression": expression,
                    "derivative": str(derivative),
                    "arc_length_integrand": str(arc_length_integrand),
                    "note": "需要提供积分区间以计算弧长值",
                }
        except Exception as e:
            return {"error": f"弧长计算出错: {str(e)}"}
