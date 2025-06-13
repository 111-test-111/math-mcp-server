# -*- coding: utf-8 -*-
"""
Calculus Computation Module
Provides comprehensive calculus computation functionalities
"""

import sympy as sp
import numpy as np
import scipy.integrate as integrate
from typing import List, Dict, Any, Optional, Union, Tuple


class CalculusCalculator:
    """Calculus calculator class providing comprehensive calculus computation functionalities"""

    def __init__(self):
        """Initialize the calculus calculator"""
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
        order: int = 1,
        method: str = "quad",
        mode: str = "symbolic",
    ) -> Dict[str, Any]:
        """
        Comprehensive calculus computation tool

        Args:
            expression: Mathematical expression string
            operation: Operation type ('derivative', 'integral', 'limit', 'series', 'critical_points', 'partial', 'gradient', 'taylor', 'arc_length')
            variable: Main variable name
            variables: List of variables (for partial derivatives, gradient, etc.)
            limits: Integration bounds or other ranges
            point: Point at which to compute
            points: Multiple points for computation
            order: Derivative order or Taylor series number of terms
            method: Computation method
            mode: Computation mode ('symbolic', 'numerical')

        Returns:
            Calculus computation result
        """
        try:
            if operation == "derivative":
                if mode == "numerical" and point is not None:
                    return self._numerical_derivative(expression, point, variable)
                else:
                    if order > 1:
                        return self._higher_order_derivatives(
                            expression, variable, order, point
                        )
                    else:
                        return self._symbolic_operations(
                            expression, "derivative", variable, limits, point
                        )
            elif operation == "integral":
                if mode == "numerical" and limits:
                    return self._numerical_integration(
                        expression, limits, variable, method
                    )
                else:
                    return self._symbolic_operations(
                        expression, "integral", variable, limits, point
                    )
            elif operation == "limit":
                return self._symbolic_operations(
                    expression, "limit", variable, limits, point
                )
            elif operation == "series":
                return self._symbolic_operations(
                    expression, "series", variable, limits, point
                )
            elif operation == "critical_points":
                return self._critical_points(expression, variable)
            elif operation == "partial" and variables:
                return self._partial_derivatives(expression, variables)
            elif operation == "gradient" and variables:
                return self._gradient(expression, variables, points)
            elif operation == "taylor":
                return self._taylor_series(expression, variable, point or 0, order)
            elif operation == "arc_length" and limits:
                return self._arc_length(expression, variable, limits)
            else:
                return {
                    "error": f"Unsupported operation type or missing necessary parameters: {operation}"
                }
        except Exception as e:
            return {"error": f"Calculus computation error: {str(e)}"}

    def _symbolic_operations(
        self,
        expression: str,
        operation: str,
        variable: str = "x",
        limits: Optional[List[float]] = None,
        point: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Symbolic calculus operations

        Args:
            expression: Mathematical expression string
            operation: Operation type ('derivative', 'integral', 'limit', 'series')
            variable: Variable name
            limits: Integration bounds or series expansion point
            point: Point for differentiation or limit

        Returns:
            Operation result
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            if operation == "derivative":
                if point is not None:
                    derivative = sp.diff(expr, var)
                    result_value = float(derivative.subs(var, point))
                    return {
                        "derivative": str(derivative),
                        "value_at_point": result_value,
                        "point": point,
                    }
                else:
                    derivative = sp.diff(expr, var)
                    return {"derivative": str(derivative)}

            elif operation == "integral":
                if limits is not None and len(limits) == 2:
                    definite_integral = sp.integrate(expr, (var, limits[0], limits[1]))
                    result_value = float(definite_integral.evalf())
                    return {
                        "definite_integral": str(definite_integral),
                        "value": result_value,
                        "limits": limits,
                    }
                else:
                    indefinite_integral = sp.integrate(expr, var)
                    return {"indefinite_integral": str(indefinite_integral)}

            elif operation == "limit":
                if point is None:
                    return {"error": "A point must be specified for limit calculation"}
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
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Symbolic calculation error: {str(e)}"}

    def _numerical_derivative(
        self, function_expr: str, point: float, variable: str = "x", h: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Numerical differentiation

        Args:
            function_expr: Function expression
            point: Point of differentiation
            variable: Variable name
            h: Step size

        Returns:
            Numerical derivative result
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(function_expr)
            func = sp.lambdify(var, expr, "numpy")

            # Central difference method
            derivative = (func(point + h) - func(point - h)) / (2 * h)

            return {
                "numerical_derivative": float(derivative),
                "point": point,
                "step_size": h,
                "method": "central_difference",
            }
        except Exception as e:
            return {"error": f"Numerical differentiation error: {str(e)}"}

    def _numerical_integration(
        self,
        function_expr: str,
        limits: List[float],
        variable: str = "x",
        method: str = "quad",
    ) -> Dict[str, Any]:
        """
        Numerical integration

        Args:
            function_expr: Function expression
            limits: Integration interval [a, b]
            variable: Variable name
            method: Integration method ('quad', 'simpson', 'trapz')

        Returns:
            Numerical integration result
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
                n = 1000  # Number of divisions
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
                n = 1000
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
                return {"error": f"Unsupported integration method: {method}"}

        except Exception as e:
            return {"error": f"Numerical integration error: {str(e)}"}

    def _higher_order_derivatives(
        self,
        expression: str,
        variable: str = "x",
        order: int = 2,
        point: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute higher order derivatives

        Args:
            expression: Mathematical expression string
            variable: Variable name
            order: Derivative order
            point: Point for differentiation (optional)

        Returns:
            Higher order derivative result
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            derivative = expr
            for _ in range(order):
                derivative = sp.diff(derivative, var)

            if point is not None:
                value_at_point = float(derivative.subs(var, point))
                return {
                    f"{order}_order_derivative": str(derivative),
                    "value_at_point": value_at_point,
                    "point": point,
                }
            else:
                return {f"{order}_order_derivative": str(derivative)}

        except Exception as e:
            return {"error": f"Higher order derivative computation error: {str(e)}"}

    def _partial_derivatives(
        self, expression: str, variables: List[str]
    ) -> Dict[str, Any]:
        """
        Partial derivatives

        Args:
            expression: Multivariable expression
            variables: List of variable names

        Returns:
            Partial derivatives result
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
            return {"error": f"Partial derivatives computation error: {str(e)}"}

    def _gradient(
        self, expression: str, variables: List[str], point: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Gradient computation

        Args:
            expression: Multivariable function expression
            variables: List of variable names
            point: Point at which to compute the gradient (optional)

        Returns:
            Gradient result
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            expr = sp.sympify(expression)

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
            return {"error": f"Gradient computation error: {str(e)}"}

    def _taylor_series(
        self, expression: str, variable: str = "x", point: float = 0, order: int = 5
    ) -> Dict[str, Any]:
        """
        Taylor series expansion

        Args:
            expression: Function expression
            variable: Variable name
            point: Expansion point
            order: Expansion order

        Returns:
            Taylor series result
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
            return {"error": f"Taylor series expansion error: {str(e)}"}

    def _critical_points(self, expression: str, variable: str = "x") -> Dict[str, Any]:
        """
        Find critical points

        Args:
            expression: Function expression
            variable: Variable name

        Returns:
            Critical points result
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            first_derivative = sp.diff(expr, var)
            critical_points = sp.solve(first_derivative, var)
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
            return {"error": f"Critical point analysis error: {str(e)}"}

    def _arc_length(
        self, expression: str, variable: str = "x", limits: List[float] = None
    ) -> Dict[str, Any]:
        """
        Arc length calculation

        Args:
            expression: Function expression y = f(x)
            variable: Variable name
            limits: Integration interval [a, b]

        Returns:
            Arc length result
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)

            derivative = sp.diff(expr, var)

            arc_length_integrand = sp.sqrt(1 + derivative**2)

            if limits is not None and len(limits) == 2:
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
                        "note": "Unable to calculate the exact value, numerical integration is required",
                    }
            else:
                return {
                    "expression": expression,
                    "derivative": str(derivative),
                    "arc_length_integrand": str(arc_length_integrand),
                    "note": "Integration interval required to compute arc length value",
                }
        except Exception as e:
            return {"error": f"Arc length calculation error: {str(e)}"}
