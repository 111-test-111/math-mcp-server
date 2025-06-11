# -*- coding: utf-8 -*-
"""
优化计算模块
提供完整丰富的优化算法功能
"""

import numpy as np
import scipy.optimize as optimize
import sympy as sp
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class OptimizationCalculator:
    """优化计算器类，提供完整的优化算法功能"""

    def __init__(self):
        """初始化优化计算器"""
        pass

    def optimization_suite_tool(
        self,
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
            if operation in ["minimize", "maximize"]:
                if method == "symbolic":
                    return self.symbolic_optimization(
                        objective_function,
                        variables,
                        constraints,
                        operation,
                        initial_guess,
                    )
                else:
                    if not initial_guess:
                        return {"error": "数值优化需要初始猜测值"}
                    return self.numerical_optimization(
                        objective_function, variables, initial_guess, method, bounds
                    )
            elif operation == "root_finding":
                equation_expr = equation or objective_function
                if len(variables) != 1:
                    return {"error": "求根操作只支持单变量"}
                return self.root_finding(
                    equation_expr,
                    variables[0],
                    initial_guess[0] if initial_guess else 1.0,
                    root_method,
                )
            elif operation == "linear_programming":
                if not lp_c:
                    return {"error": "线性规划需要目标函数系数"}
                return self.linear_programming(
                    lp_c, lp_A_ub, lp_b_ub, lp_A_eq, lp_b_eq, bounds
                )
            elif operation == "least_squares":
                if not initial_guess:
                    return {"error": "最小二乘需要初始猜测值"}
                return self.least_squares(objective_function, variables, initial_guess)
            elif operation == "constrained":
                if not constraints or not initial_guess:
                    return {"error": "约束优化需要约束条件和初始猜测值"}
                constraint_dicts = [{"type": "eq", "fun": c} for c in constraints]
                return self.constrained_optimization(
                    objective_function,
                    variables,
                    constraint_dicts,
                    initial_guess,
                    method,
                )
            elif operation == "global":
                if not bounds:
                    return {"error": "全局优化需要变量边界"}
                return self.global_optimization(
                    objective_function, variables, bounds, method
                )
            else:
                return {"error": f"不支持的操作类型: {operation}"}
        except Exception as e:
            return {"error": f"优化计算出错: {str(e)}"}

    def symbolic_optimization(
        self,
        objective_function: str,
        variables: List[str],
        constraints: Optional[List[str]] = None,
        method: str = "minimize",
        initial_guess: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        符号优化求解

        Args:
            objective_function: 目标函数表达式
            variables: 变量列表
            constraints: 约束条件列表
            method: 优化方法 ('minimize', 'maximize')
            initial_guess: 初始猜测值

        Returns:
            优化结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)

            # 如果是最大化问题，转换为最小化
            if method == "maximize":
                obj_expr = -obj_expr

            # 计算梯度
            gradient = [sp.diff(obj_expr, var) for var in vars_symbols]

            # 求解临界点（梯度为零的点）
            critical_points = sp.solve(gradient, vars_symbols)

            # 计算Hessian矩阵
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

            # 分析临界点
            if critical_points:
                if isinstance(critical_points, dict):
                    critical_points = [critical_points]

                point_analysis = []
                for point in critical_points:
                    if isinstance(point, dict):
                        point_values = [float(point[var]) for var in vars_symbols]

                        # 计算该点的函数值
                        substitutions = point
                        func_value = float(obj_expr.subs(substitutions))

                        # 如果是最大化问题，恢复原始函数值
                        if method == "maximize":
                            func_value = -func_value

                        # 计算Hessian矩阵的数值
                        hessian_numerical = []
                        for row in hessian:
                            hessian_row = []
                            for elem in row:
                                hessian_row.append(float(elem.subs(substitutions)))
                            hessian_numerical.append(hessian_row)

                        # 判断点的性质（通过Hessian矩阵的特征值）
                        hessian_matrix = np.array(hessian_numerical)
                        eigenvalues = np.linalg.eigvals(hessian_matrix)

                        if all(eig > 0 for eig in eigenvalues):
                            point_type = (
                                "local_minimum"
                                if method == "minimize"
                                else "local_maximum"
                            )
                        elif all(eig < 0 for eig in eigenvalues):
                            point_type = (
                                "local_maximum"
                                if method == "minimize"
                                else "local_minimum"
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
            return {"error": f"符号优化出错: {str(e)}"}

    def numerical_optimization(
        self,
        objective_function: str,
        variables: List[str],
        initial_guess: List[float],
        method: str = "BFGS",
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        数值优化求解

        Args:
            objective_function: 目标函数表达式
            variables: 变量列表
            initial_guess: 初始猜测值
            method: 优化方法 ('BFGS', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead')
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            优化结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)

            # 创建数值函数
            obj_func = sp.lambdify(vars_symbols, obj_expr, "numpy")

            # 计算梯度函数
            gradient_exprs = [sp.diff(obj_expr, var) for var in vars_symbols]
            gradient_func = sp.lambdify(vars_symbols, gradient_exprs, "numpy")

            def objective(x):
                return float(obj_func(*x))

            def gradient(x):
                grad = gradient_func(*x)
                if isinstance(grad, (int, float)):
                    return np.array([grad])
                return np.array(grad)

            # 执行优化
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
            return {"error": f"数值优化出错: {str(e)}"}

    def linear_programming(
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
        线性规划求解

        Args:
            c: 目标函数系数
            A_ub: 不等式约束矩阵
            b_ub: 不等式约束右端
            A_eq: 等式约束矩阵
            b_eq: 等式约束右端
            bounds: 变量边界
            method: 求解方法

        Returns:
            线性规划结果
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
            return {"error": f"线性规划求解出错: {str(e)}"}

    def root_finding(
        self,
        function_expr: str,
        variable: str = "x",
        initial_guess: float = 1.0,
        method: str = "fsolve",
    ) -> Dict[str, Any]:
        """
        方程求根

        Args:
            function_expr: 函数表达式
            variable: 变量名
            initial_guess: 初始猜测值
            method: 求根方法 ('fsolve', 'newton', 'bisect')

        Returns:
            求根结果
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
                # 计算导数
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
                # 对于二分法，需要提供区间
                # 这里使用初始猜测值附近的区间
                a = initial_guess - 1
                b = initial_guess + 1

                # 确保f(a)和f(b)异号
                fa, fb = func(a), func(b)
                if fa * fb > 0:
                    # 扩展搜索区间
                    for i in range(10):
                        a -= 1
                        b += 1
                        fa, fb = func(a), func(b)
                        if fa * fb < 0:
                            break

                if fa * fb > 0:
                    return {"error": "无法找到合适的区间进行二分法求根"}

                root = optimize.bisect(func, a, b)

                return {
                    "function": function_expr,
                    "method": method,
                    "root": float(root),
                    "interval": [a, b],
                }

            else:
                return {"error": f"不支持的求根方法: {method}"}

        except Exception as e:
            return {"error": f"求根计算出错: {str(e)}"}

    def least_squares(
        self,
        residual_function: str,
        variables: List[str],
        initial_guess: List[float],
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        最小二乘拟合

        Args:
            residual_function: 残差函数表达式
            variables: 变量列表
            initial_guess: 初始猜测值
            x_data: x数据点
            y_data: y数据点

        Returns:
            最小二乘结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]

            # 如果提供了数据点，构建残差函数
            if x_data is not None and y_data is not None:
                x_symbol = sp.Symbol("x")
                # 假设residual_function是关于x的函数，包含待拟合参数
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
                # 直接使用提供的残差函数
                expr = sp.sympify(residual_function)
                residual_func = sp.lambdify(vars_symbols, expr, "numpy")

                def residual(params):
                    return residual_func(*params)

            # 执行最小二乘优化
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
            return {"error": f"最小二乘拟合出错: {str(e)}"}

    def constrained_optimization(
        self,
        objective_function: str,
        variables: List[str],
        constraints: List[Dict[str, str]],
        initial_guess: List[float],
        method: str = "SLSQP",
    ) -> Dict[str, Any]:
        """
        约束优化

        Args:
            objective_function: 目标函数表达式
            variables: 变量列表
            constraints: 约束条件列表，格式为[{'type': 'eq'/'ineq', 'fun': 'expression'}]
            initial_guess: 初始猜测值
            method: 优化方法

        Returns:
            约束优化结果
        """
        try:
            vars_symbols = [sp.Symbol(var) for var in variables]
            obj_expr = sp.sympify(objective_function)
            obj_func = sp.lambdify(vars_symbols, obj_expr, "numpy")

            def objective(x):
                return float(obj_func(*x))

            # 构建约束条件
            constraint_funcs = []
            for constraint in constraints:
                constraint_expr = sp.sympify(constraint["fun"])
                constraint_func = sp.lambdify(vars_symbols, constraint_expr, "numpy")

                constraint_funcs.append(
                    {
                        "type": constraint["type"],
                        "fun": lambda x, cf=constraint_func: float(cf(*x)),
                    }
                )

            # 执行约束优化
            result = optimize.minimize(
                objective, initial_guess, method=method, constraints=constraint_funcs
            )

            return {
                "objective_function": objective_function,
                "variables": variables,
                "constraints": constraints,
                "method": method,
                "success": result.success,
                "optimal_point": result.x.tolist(),
                "optimal_value": float(result.fun),
                "constraint_violations": (
                    result.maxcv if hasattr(result, "maxcv") else None
                ),
                "message": result.message,
            }

        except Exception as e:
            return {"error": f"约束优化出错: {str(e)}"}

    def global_optimization(
        self,
        objective_function: str,
        variables: List[str],
        bounds: List[Tuple[float, float]],
        method: str = "differential_evolution",
    ) -> Dict[str, Any]:
        """
        全局优化

        Args:
            objective_function: 目标函数表达式
            variables: 变量列表
            bounds: 变量边界
            method: 全局优化方法 ('differential_evolution', 'basinhopping')

        Returns:
            全局优化结果
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
                # 对于basinhopping，需要一个初始点
                initial_guess = [(b[0] + b[1]) / 2 for b in bounds]
                result = optimize.basinhopping(objective, initial_guess)
            else:
                return {"error": f"不支持的全局优化方法: {method}"}

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
            return {"error": f"全局优化出错: {str(e)}"}
