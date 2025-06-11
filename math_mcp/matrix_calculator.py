# -*- coding: utf-8 -*-
"""
矩阵计算模块
提供完整丰富的矩阵运算功能
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class MatrixCalculator:
    """矩阵计算器类，提供完整的矩阵运算功能"""

    def __init__(self):
        """初始化矩阵计算器"""
        pass

    def matrix_calculator_tool(
        self,
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
            if operation == "basic" and method:
                return self.basic_operations(method, matrix_a, matrix_b)
            elif operation == "eigenvalues":
                return self.eigenvalues_eigenvectors(matrix_a)
            elif operation == "svd":
                return self.singular_value_decomposition(matrix_a)
            elif operation == "decomposition" and method:
                if method == "qr":
                    return self.qr_decomposition(matrix_a)
                elif method == "lu":
                    return self.lu_decomposition(matrix_a)
                else:
                    return {"error": f"不支持的分解方法: {method}"}
            elif operation == "properties":
                prop_type = method or property_type or "rank"
                if prop_type == "rank":
                    return self.matrix_rank(matrix_a)
                elif prop_type == "trace":
                    return self.matrix_trace(matrix_a)
                elif prop_type == "condition_number":
                    return self.matrix_condition_number(matrix_a)
                elif prop_type == "norm":
                    return self.matrix_norm(matrix_a)
                else:
                    return {"error": f"不支持的属性类型: {prop_type}"}
            elif operation == "power" and power is not None:
                return self.matrix_power(matrix_a, power)
            elif operation == "exponential":
                return self.matrix_exponential(matrix_a)
            elif operation == "solve" and matrix_b:
                # matrix_b在这里作为常数向量
                constants = (
                    [row[0] for row in matrix_b]
                    if isinstance(matrix_b[0], list)
                    else matrix_b
                )
                return self.solve_linear_system(matrix_a, constants)
            else:
                return {"error": f"不支持的操作或缺少必要参数: {operation}"}
        except Exception as e:
            return {"error": f"矩阵计算出错: {str(e)}"}

    def basic_operations(
        self,
        operation: str,
        matrix_a: List[List[float]],
        matrix_b: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        基础矩阵运算

        Args:
            operation: 运算类型 ('add', 'subtract', 'multiply', 'transpose', 'determinant', 'inverse')
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵（某些运算需要）

        Returns:
            运算结果
        """
        try:
            A = np.array(matrix_a)

            if operation == "transpose":
                result = A.T.tolist()
                return {"result": result, "shape": A.T.shape}

            elif operation == "determinant":
                if A.shape[0] != A.shape[1]:
                    return {"error": "矩阵必须是方阵才能计算行列式"}
                det = np.linalg.det(A)
                return {"determinant": float(det)}

            elif operation == "inverse":
                if A.shape[0] != A.shape[1]:
                    return {"error": "矩阵必须是方阵才能求逆"}
                try:
                    inv_A = np.linalg.inv(A)
                    return {"result": inv_A.tolist(), "shape": inv_A.shape}
                except np.linalg.LinAlgError:
                    return {"error": "矩阵不可逆"}

            elif operation in ["add", "subtract", "multiply"]:
                if matrix_b is None:
                    return {"error": f"{operation} 运算需要两个矩阵"}

                B = np.array(matrix_b)

                if operation == "add":
                    if A.shape != B.shape:
                        return {"error": "矩阵加法要求两矩阵形状相同"}
                    result = (A + B).tolist()

                elif operation == "subtract":
                    if A.shape != B.shape:
                        return {"error": "矩阵减法要求两矩阵形状相同"}
                    result = (A - B).tolist()

                elif operation == "multiply":
                    if A.shape[1] != B.shape[0]:
                        return {
                            "error": "矩阵乘法要求第一个矩阵的列数等于第二个矩阵的行数"
                        }
                    result = (A @ B).tolist()

                return {"result": result, "shape": np.array(result).shape}

            else:
                return {"error": f"不支持的运算类型: {operation}"}

        except Exception as e:
            return {"error": f"计算出错: {str(e)}"}

    def eigenvalues_eigenvectors(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        计算矩阵的特征值和特征向量

        Args:
            matrix: 输入矩阵

        Returns:
            特征值和特征向量
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "矩阵必须是方阵才能计算特征值"}

            eigenvals, eigenvecs = np.linalg.eig(A)
            return {
                "eigenvalues": eigenvals.tolist(),
                "eigenvectors": eigenvecs.tolist(),
            }
        except Exception as e:
            return {"error": f"特征值计算出错: {str(e)}"}

    def singular_value_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        奇异值分解

        Args:
            matrix: 输入矩阵

        Returns:
            SVD分解结果
        """
        try:
            A = np.array(matrix)
            U, s, Vt = np.linalg.svd(A)
            return {"U": U.tolist(), "singular_values": s.tolist(), "Vt": Vt.tolist()}
        except Exception as e:
            return {"error": f"SVD分解出错: {str(e)}"}

    def qr_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        QR分解

        Args:
            matrix: 输入矩阵

        Returns:
            QR分解结果
        """
        try:
            A = np.array(matrix)
            Q, R = np.linalg.qr(A)
            return {"Q": Q.tolist(), "R": R.tolist()}
        except Exception as e:
            return {"error": f"QR分解出错: {str(e)}"}

    def lu_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        LU分解

        Args:
            matrix: 输入矩阵

        Returns:
            LU分解结果
        """
        try:
            from scipy.linalg import lu

            A = np.array(matrix)
            P, L, U = lu(A)
            return {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}
        except Exception as e:
            return {"error": f"LU分解出错: {str(e)}"}

    def matrix_rank(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        计算矩阵的秩

        Args:
            matrix: 输入矩阵

        Returns:
            矩阵的秩
        """
        try:
            A = np.array(matrix)
            rank = np.linalg.matrix_rank(A)
            return {"rank": int(rank)}
        except Exception as e:
            return {"error": f"计算矩阵秩出错: {str(e)}"}

    def matrix_norm(
        self, matrix: List[List[float]], norm_type: str = "frobenius"
    ) -> Dict[str, Any]:
        """
        计算矩阵范数

        Args:
            matrix: 输入矩阵
            norm_type: 范数类型 ('frobenius', 'nuclear', '1', '2', 'inf')

        Returns:
            矩阵范数
        """
        try:
            A = np.array(matrix)
            if norm_type == "frobenius":
                norm_val = np.linalg.norm(A, "fro")
            elif norm_type == "nuclear":
                norm_val = np.linalg.norm(A, "nuc")
            elif norm_type in ["1", "2", "inf"]:
                norm_val = np.linalg.norm(A, ord=norm_type)
            else:
                return {"error": f"不支持的范数类型: {norm_type}"}

            return {"norm": float(norm_val), "norm_type": norm_type}
        except Exception as e:
            return {"error": f"计算矩阵范数出错: {str(e)}"}

    def matrix_trace(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        计算矩阵的迹

        Args:
            matrix: 输入矩阵

        Returns:
            矩阵的迹
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "矩阵必须是方阵才能计算迹"}

            trace_val = np.trace(A)
            return {"trace": float(trace_val)}
        except Exception as e:
            return {"error": f"计算矩阵迹出错: {str(e)}"}

    def matrix_condition_number(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        计算矩阵的条件数

        Args:
            matrix: 输入矩阵

        Returns:
            矩阵的条件数
        """
        try:
            A = np.array(matrix)
            cond_num = np.linalg.cond(A)
            return {"condition_number": float(cond_num)}
        except Exception as e:
            return {"error": f"计算条件数出错: {str(e)}"}

    def matrix_power(self, matrix: List[List[float]], power: int) -> Dict[str, Any]:
        """
        计算矩阵的幂

        Args:
            matrix: 输入矩阵
            power: 幂次

        Returns:
            矩阵的幂
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "矩阵必须是方阵才能计算幂"}

            result = np.linalg.matrix_power(A, power)
            return {"result": result.tolist(), "power": power}
        except Exception as e:
            return {"error": f"计算矩阵幂出错: {str(e)}"}

    def matrix_exponential(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        计算矩阵指数

        Args:
            matrix: 输入矩阵

        Returns:
            矩阵指数
        """
        try:
            from scipy.linalg import expm

            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "矩阵必须是方阵才能计算矩阵指数"}

            result = expm(A)
            return {"result": result.tolist()}
        except Exception as e:
            return {"error": f"计算矩阵指数出错: {str(e)}"}

    def solve_linear_system(
        self, coefficients: List[List[float]], constants: List[float]
    ) -> Dict[str, Any]:
        """
        求解线性方程组 Ax = b

        Args:
            coefficients: 系数矩阵A
            constants: 常数向量b

        Returns:
            方程组的解
        """
        try:
            A = np.array(coefficients)
            b = np.array(constants)

            if A.shape[0] != len(constants):
                return {"error": "系数矩阵的行数必须等于常数向量的长度"}

            solution = np.linalg.solve(A, b)
            return {"solution": solution.tolist()}
        except np.linalg.LinAlgError:
            return {"error": "线性方程组无解或有无穷多解"}
        except Exception as e:
            return {"error": f"求解线性方程组出错: {str(e)}"}
