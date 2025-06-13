# -*- coding: utf-8 -*-
"""
Optimization Calculation Module
Provides comprehensive and rich optimization algorithm functionality
"""

import numpy as np
import scipy.optimize as optimize
import sympy as sp
from typing import List, Dict, Any, Optional, Union, Tuple


class OptimizationCalculator:
    """Optimization calculator class, providing comprehensive optimization algorithm functionality"""

    def __init__(self):
        """Initialize optimization calculator"""
        pass

    def optimization_suite_tool(
        self,
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
        """
        Comprehensive optimization calculation tool

        Args:
            objective_function: Objective function expression
            variables: Variable list
            operation: Operation type ('minimize', 'maximize', 'root_finding', 'linear_programming', 'least_squares', 'constrained', 'global')
            method: Calculation method
            initial_guess: Initial guess values
            bounds: Variable bounds
            constraints: Constraint conditions
            equation: Equation (used for root finding)
            root_method: Root finding method
            lp_c: Linear programming objective function coefficients
            lp_A_ub: Linear programming inequality constraint matrix
            lp_b_ub: Linear programming inequality constraint vector
            lp_A_eq: Linear programming equality constraint matrix
            lp_b_eq: Linear programming equality constraint vector

        Returns:
            Optimization calculation results
        """
        try:
            # Automatically select appropriate method
            if method == "auto":
                if operation == "constrained":
                    method = "SLSQP"
                elif operation == "global":
                    method = "differential_evolution"
                elif operation in ["minimize", "maximize"]:
                    method = "symbolic" if not initial_guess else "BFGS"
                elif operation == "linear_programming":
                    method = "highs"
                else:
                    method = "BFGS"

            if operation in ["minimize", "maximize"]:
                if method == "symbolic":
                    return self._symbolic_optimization(
                        objective_function,
                        variables,
                        constraints,
                        operation,
                        initial_guess,
                    )
                else:
                    if not initial_guess:
                        # Provide default initial guess values for numerical optimization
                        initial_guess = [0.0] * len(variables)
                    return self._numerical_optimization(
                        objective_function, variables, initial_guess, method, bounds
                    )
            elif operation == "root_finding":
                equation_expr = equation or objective_function
                if len(variables) != 1:
                    return {
                        "error": "Root finding operation only supports single variable"
                    }
                return self._root_finding(
                    equation_expr,
                    variables[0],
                    initial_guess[0] if initial_guess else 1.0,
                    root_method,
                )
            elif operation == "linear_programming":
                if not lp_c:
                    return {
                        "error": "Linear programming requires objective function coefficients"
                    }
                return self._linear_programming(
                    lp_c, lp_A_ub, lp_b_ub, lp_A_eq, lp_b_eq, bounds
                )
            elif operation == "least_squares":
                if not initial_guess:
                    initial_guess = [1.0] * len(variables)
                return self._least_squares(objective_function, variables, initial_guess)
            elif operation == "constrained":
                if not constraints:
                    return {
                        "error": "Constrained optimization requires constraint conditions"
                    }
                if not initial_guess:
                    # Provide default initial guess values
                    initial_guess = [0.0] * len(variables)
                return self._constrained_optimization(
                    objective_function,
                    variables,
                    constraints,
                    initial_guess,
                    method,
                )
            elif operation == "global":
                if not bounds:
                    return {"error": "Global optimization requires variable bounds"}
                return self._global_optimization(
                    objective_function, variables, bounds, method
                )
            else:
                return {"error": f"Unsupported operation type: {operation}"}
        except Exception as e:
            return {"error": f"Optimization calculation error: {str(e)}"}

    def _symbolic_optimization(
        self,
        objective_function: str,
        variables: List[str],
        constraints: Optional[List[str]] = None,
        method: str = "minimize",
        initial_guess: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Symbolic optimization solving

        Args:
            objective_function: Objective function expression
            variables: Variable list
            constraints: Constraint condition list
            method: Optimization method ('minimize', 'maximize')
            initial_guess: Initial guess values

        Returns:
            Optimization results
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)

            # Convert maximization problem to minimization
            if method == "maximize":
                obj_expr = -obj_expr

            # Calculate gradient
            gradient = [sp.diff(obj_expr, var) for var in vars_symbols]

            # Solve critical points (points where gradient equals zero)
            critical_points = sp.solve(gradient, vars_symbols)

            # Calculate Hessian matrix
            hessian = []
            for i, var1 in enumerate(vars_symbols):
                hessian_row = []
                for j, var2 in enumerate(vars_symbols):
                    second_deriv = sp.diff(obj_expr, var1, var2)
                    hessian_row.append(second_deriv)
                hessian.append(hessian_row)

            result = {
                "objective_function": objective_function,
                "variables": variables,
                "method": method,
                "gradient": [str(g) for g in gradient],
                "hessian": [[str(h) for h in row] for row in hessian],
            }

            # Analyze critical points
            if critical_points:
                if isinstance(critical_points, dict):
                    critical_points = [critical_points]

                point_analysis = []
                for point in critical_points:
                    substitutions = {}
                    point_values = []
                    if isinstance(point, dict):
                        substitutions = point
                        point_values = [float(point[var]) for var in vars_symbols]
                    elif isinstance(point, (list, tuple)):
                        substitutions = dict(zip(vars_symbols, point))
                        point_values = [float(v) for v in point]
                    else:
                        # Single variable case
                        substitutions = {vars_symbols[0]: point}
                        point_values = [float(point)]

                    # Calculate function value at this point
                    func_value = float(obj_expr.subs(substitutions))

                    # Restore original function value for maximization problem
                    if method == "maximize":
                        func_value = -func_value

                    # Calculate numerical Hessian matrix
                    hessian_numerical = []
                    for row in hessian:
                        hessian_row = []
                        for elem in row:
                            hessian_row.append(float(elem.subs(substitutions)))
                        hessian_numerical.append(hessian_row)

                    # Determine point nature (through eigenvalues of Hessian matrix)
                    hessian_matrix = np.array(hessian_numerical)
                    eigenvalues = np.linalg.eigvals(hessian_matrix)

                    if all(eig > 0 for eig in eigenvalues):
                        point_type = (
                            "local_minimum" if method == "minimize" else "local_maximum"
                        )
                    elif all(eig < 0 for eig in eigenvalues):
                        point_type = (
                            "local_maximum" if method == "minimize" else "local_minimum"
                        )
                    else:
                        point_type = "saddle_point"

                    point_analysis.append(
                        {
                            "point": point_values,
                            "function_value": func_value,
                            "type": point_type,
                            "eigenvalues": eigenvalues.tolist(),
                        }
                    )

                result["critical_points"] = point_analysis

            return result

        except Exception as e:
            return {"error": f"Symbolic optimization error: {str(e)}"}

    def _numerical_optimization(
        self,
        objective_function: str,
        variables: List[str],
        initial_guess: List[float],
        method: str = "BFGS",
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Numerical optimization solving

        Args:
            objective_function: Objective function expression
            variables: Variable list
            initial_guess: Initial guess values
            method: Optimization method ('BFGS', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead')
            bounds: Variable bounds
            constraints: Constraint conditions

        Returns:
            Optimization results
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)

            # Create numerical function
            obj_func = sp.lambdify(vars_symbols, obj_expr, "numpy")

            # Calculate gradient function
            gradient_exprs = [sp.diff(obj_expr, var) for var in vars_symbols]
            gradient_func = sp.lambdify(vars_symbols, gradient_exprs, "numpy")

            def objective(x):
                return float(obj_func(*x))

            def gradient(x):
                grad = gradient_func(*x)
                if isinstance(grad, (int, float)):
                    return np.array([grad])
                return np.array(grad)

            # Execute optimization
            if method in ["L-BFGS-B", "SLSQP"]:
                result = optimize.minimize(
                    objective,
                    initial_guess,
                    method=method,
                    jac=gradient,
                    bounds=bounds,
                    constraints=constraints,
                )
            else:
                result = optimize.minimize(
                    objective, initial_guess, method=method, jac=gradient
                )

            return {
                "objective_function": objective_function,
                "variables": variables,
                "method": method,
                "success": result.success,
                "optimal_point": result.x.tolist(),
                "optimal_value": float(result.fun),
                "iterations": getattr(result, "nit", None),
                "function_evaluations": result.nfev,
                "message": result.message,
            }

        except Exception as e:
            return {"error": f"Numerical optimization error: {str(e)}"}

    def _linear_programming(
        self,
        c: List[float],
        A_ub: Optional[List[List[float]]] = None,
        b_ub: Optional[List[float]] = None,
        A_eq: Optional[List[List[float]]] = None,
        b_eq: Optional[List[float]] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        method: str = "highs",
    ) -> Dict[str, Any]:
        """
        Linear programming solving

        Args:
            c: Objective function coefficients
            A_ub: Inequality constraint matrix
            b_ub: Inequality constraint right-hand side
            A_eq: Equality constraint matrix
            b_eq: Equality constraint right-hand side
            bounds: Variable bounds
            method: Solving method

        Returns:
            Linear programming results
        """
        try:
            result = optimize.linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
            )

            return {
                "success": result.success,
                "optimal_point": result.x.tolist() if result.x is not None else None,
                "optimal_value": float(result.fun) if result.fun is not None else None,
                "slack": (
                    result.slack.tolist()
                    if hasattr(result, "slack") and result.slack is not None
                    else None
                ),
                "message": result.message,
                "method": method,
            }

        except Exception as e:
            return {"error": f"Linear programming solving error: {str(e)}"}

    def _root_finding(
        self,
        function_expr: str,
        variable: str = "x",
        initial_guess: float = 1.0,
        method: str = "fsolve",
    ) -> Dict[str, Any]:
        """
        Equation root finding

        Args:
            function_expr: Function expression
            variable: Variable name
            initial_guess: Initial guess value
            method: Root finding method ('fsolve', 'newton', 'bisect')

        Returns:
            Root finding results
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(function_expr)
            func = sp.lambdify(var, expr, "numpy")

            if method == "fsolve":
                result = optimize.fsolve(func, initial_guess, full_output=True)
                roots = result[0]
                info = result[1]

                return {
                    "function": function_expr,
                    "method": method,
                    "roots": roots.tolist(),
                    "function_calls": info["nfev"],
                    "success": "converged" in str(info),
                }

            elif method == "newton":
                # Calculate derivative
                derivative = sp.diff(expr, var)
                dfunc = sp.lambdify(var, derivative, "numpy")

                root = optimize.newton(func, initial_guess, fprime=dfunc)

                return {
                    "function": function_expr,
                    "method": method,
                    "root": float(root),
                    "initial_guess": initial_guess,
                }

            elif method == "bisect":
                # For bisection method, need to provide interval
                # Here use interval around initial guess value
                a = initial_guess - 1
                b = initial_guess + 1

                # Ensure f(a) and f(b) have opposite signs
                fa, fb = func(a), func(b)
                if fa * fb > 0:
                    # Expand search interval
                    for i in range(10):
                        a -= 1
                        b += 1
                        fa, fb = func(a), func(b)
                        if fa * fb < 0:
                            break

                if fa * fb > 0:
                    return {
                        "error": "Cannot find suitable interval for bisection method root finding"
                    }

                root = optimize.bisect(func, a, b)

                return {
                    "function": function_expr,
                    "method": method,
                    "root": float(root),
                    "interval": [a, b],
                }

            else:
                return {"error": f"Unsupported root finding method: {method}"}

        except Exception as e:
            return {"error": f"Root finding calculation error: {str(e)}"}

    def _least_squares(
        self,
        residual_function: str,
        variables: List[str],
        initial_guess: List[float],
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Least squares fitting

        Args:
            residual_function: Residual function expression
            variables: Variable list
            initial_guess: Initial guess values
            x_data: x data points
            y_data: y data points

        Returns:
            Least squares results
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]

            # If data points are provided, construct residual function
            if x_data is not None and y_data is not None:
                x_symbol = sp.Symbol("x")
                # Assume residual_function is a function of x containing parameters to be fitted
                expr = sp.sympify(residual_function)

                def residual(params):
                    substitutions = dict(zip(vars_symbols, params))
                    residuals = []
                    for xi, yi in zip(x_data, y_data):
                        substitutions[x_symbol] = xi
                        predicted = float(expr.subs(substitutions))
                        residuals.append(predicted - yi)
                    return np.array(residuals)

            else:
                # Directly use provided residual function
                expr = sp.sympify(residual_function)
                residual_func = sp.lambdify(vars_symbols, expr, "numpy")

                def residual(params):
                    return residual_func(*params)

            # Execute least squares optimization
            result = optimize.least_squares(residual, initial_guess)

            return {
                "residual_function": residual_function,
                "variables": variables,
                "success": result.success,
                "optimal_parameters": result.x.tolist(),
                "residual_norm": float(result.cost),
                "function_evaluations": result.nfev,
                "optimality": float(result.optimality),
                "message": result.message,
            }

        except Exception as e:
            return {"error": f"Least squares fitting error: {str(e)}"}

    def _constrained_optimization(
        self,
        objective_function: str,
        variables: List[str],
        constraints: List[Dict[str, str]],
        initial_guess: List[float],
        method: str = "SLSQP",
    ) -> Dict[str, Any]:
        """
        Constrained numerical optimization

        Args:
            objective_function: Objective function expression
            variables: Variable list
            constraints: Constraint condition list, format: [{'type': 'eq'/'ineq', 'fun': 'expression'}]
            initial_guess: Initial guess values
            method: Optimization method

        Returns:
            Optimization results
        """
        try:
            # Convert objective function
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)

            def objective(x):
                substitutions = dict(zip(vars_symbols, x))
                return float(obj_expr.subs(substitutions))

            # Convert constraints
            constraint_funcs = []
            for constr in constraints:
                constr_expr = sp.sympify(constr["fun"])

                def constraint_func(x, expr=constr_expr):
                    substitutions = dict(zip(vars_symbols, x))
                    return float(expr.subs(substitutions))

                constraint_funcs.append(
                    {"type": constr["type"], "fun": constraint_func}
                )

            # Execute optimization
            result = optimize.minimize(
                objective,
                initial_guess,
                method=method,
                constraints=constraint_funcs,
            )

            return {
                "solution": result.x.tolist(),
                "objective_value": float(result.fun),
                "success": bool(result.success),
                "message": result.message,
            }

        except Exception as e:
            return {"error": f"Constrained optimization error: {str(e)}"}

    def _global_optimization(
        self,
        objective_function: str,
        variables: List[str],
        bounds: List[Tuple[float, float]],
        method: str = "differential_evolution",
    ) -> Dict[str, Any]:
        """
        Global optimization

        Args:
            objective_function: Objective function expression
            variables: Variable list
            bounds: Variable bounds
            method: Global optimization method ('differential_evolution', 'basinhopping')

        Returns:
            Global optimization results
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)
            obj_func = sp.lambdify(vars_symbols, obj_expr, "numpy")

            def objective(x):
                return float(obj_func(*x))

            if method == "differential_evolution":
                result = optimize.differential_evolution(objective, bounds)
            elif method == "basinhopping":
                # For basinhopping, need an initial point
                initial_guess = [(b[0] + b[1]) / 2 for b in bounds]
                result = optimize.basinhopping(objective, initial_guess)
            else:
                return {"error": f"Unsupported global optimization method: {method}"}

            return {
                "objective_function": objective_function,
                "variables": variables,
                "bounds": bounds,
                "method": method,
                "success": result.success,
                "optimal_point": result.x.tolist(),
                "optimal_value": float(result.fun),
                "function_evaluations": result.nfev,
                "message": getattr(result, "message", "Global optimization completed"),
            }

        except Exception as e:
            return {"error": f"Global optimization error: {str(e)}"}
