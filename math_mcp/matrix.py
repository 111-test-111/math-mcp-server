# -*- coding: utf-8 -*-
"""
Matrix calculation module
Provides comprehensive and rich matrix operation functionality
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


class MatrixCalculator:
    """Matrix calculator class providing comprehensive matrix operation functionality"""

    def __init__(self):
        """Initialize matrix calculator"""
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
        Comprehensive matrix calculation tool - combines all matrix-related operations

        Args:
            operation: Operation type ('basic', 'decomposition', 'eigenvalues', 'svd', 'properties', 'power', 'exponential', 'solve')
            matrix_a: First matrix
            matrix_b: Second matrix (required for certain operations)
            method: Method type (for basic operations: 'add', 'subtract', 'multiply', 'transpose', 'determinant', 'inverse';
                              for decomposition: 'qr', 'lu';
                              for properties: 'rank', 'trace', 'condition_number', 'norm')
            power: Matrix power (for power operations)
            property_type: Property type (for backward compatibility)

        Returns:
            Calculation result
        """
        try:
            if operation == "basic" and method:
                return self._basic_operations(method, matrix_a, matrix_b)
            elif operation == "eigenvalues":
                return self._eigenvalues_eigenvectors(matrix_a)
            elif operation == "svd":
                return self._singular_value_decomposition(matrix_a)
            elif operation == "decomposition" and method:
                if method == "qr":
                    return self._qr_decomposition(matrix_a)
                elif method == "lu":
                    return self._lu_decomposition(matrix_a)
                else:
                    return {"error": f"Unsupported decomposition method: {method}"}
            elif operation == "properties":
                prop_type = method or property_type or "rank"
                if prop_type == "rank":
                    return self._matrix_rank(matrix_a)
                elif prop_type == "trace":
                    return self._matrix_trace(matrix_a)
                elif prop_type == "condition_number":
                    return self._matrix_condition_number(matrix_a)
                elif prop_type == "norm":
                    # If property_type is not provided, default to 'frobenius'
                    norm_selection = property_type or "frobenius"
                    return self._matrix_norm(matrix_a, norm_type=norm_selection)
                else:
                    return {"error": f"Unsupported property type: {prop_type}"}
            elif operation == "power" and power is not None:
                return self._matrix_power(matrix_a, power)
            elif operation == "exponential":
                return self._matrix_exponential(matrix_a)
            elif operation == "solve" and matrix_b:
                # matrix_b here serves as the constant vector
                constants = (
                    [row[0] for row in matrix_b]
                    if isinstance(matrix_b[0], list)
                    else matrix_b
                )
                return self._solve_linear_system(matrix_a, constants)
            else:
                return {
                    "error": f"Unsupported operation or missing required parameters: {operation}"
                }
        except Exception as e:
            return {"error": f"Matrix calculation error: {str(e)}"}

    def _basic_operations(
        self,
        operation: str,
        matrix_a: List[List[float]],
        matrix_b: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Basic matrix operations

        Args:
            operation: Operation type ('add', 'subtract', 'multiply', 'transpose', 'determinant', 'inverse')
            matrix_a: First matrix
            matrix_b: Second matrix (required for certain operations)

        Returns:
            Operation result
        """
        try:
            # Check if matrix is empty
            if not matrix_a or len(matrix_a) == 0 or len(matrix_a[0]) == 0:
                return {"error": "Matrix cannot be empty"}

            # Check if matrix rows have consistent length
            if not all(len(row) == len(matrix_a[0]) for row in matrix_a):
                return {"error": "Matrix row lengths must be consistent"}

            A = np.array(matrix_a)

            if operation == "transpose":
                result = A.T.tolist()
                return {"result": result, "shape": A.T.shape}

            elif operation == "determinant":
                if A.shape[0] != A.shape[1]:
                    return {"error": "Matrix must be square to calculate determinant"}
                det = np.linalg.det(A)
                return {"determinant": float(det)}

            elif operation == "inverse":
                if A.shape[0] != A.shape[1]:
                    return {"error": "Matrix must be square to calculate inverse"}
                try:
                    inv_A = np.linalg.inv(A)
                    return {"result": inv_A.tolist(), "shape": inv_A.shape}
                except np.linalg.LinAlgError:
                    return {"error": "Matrix is not invertible"}

            elif operation in ["add", "subtract", "multiply"]:
                if matrix_b is None:
                    return {"error": f"{operation} operation requires two matrices"}

                B = np.array(matrix_b)

                if operation == "add":
                    if A.shape != B.shape:
                        return {
                            "error": "Matrix addition requires matrices of the same shape"
                        }
                    result = (A + B).tolist()

                elif operation == "subtract":
                    if A.shape != B.shape:
                        return {
                            "error": "Matrix subtraction requires matrices of the same shape"
                        }
                    result = (A - B).tolist()

                elif operation == "multiply":
                    if A.shape[1] != B.shape[0]:
                        return {
                            "error": "Matrix multiplication requires the number of columns in the first matrix to equal the number of rows in the second matrix"
                        }
                    result = (A @ B).tolist()

                return {"result": result, "shape": np.array(result).shape}

            else:
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}

    def _eigenvalues_eigenvectors(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate eigenvalues and eigenvectors of a matrix

        Args:
            matrix: Input matrix

        Returns:
            Eigenvalues and eigenvectors
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "Matrix must be square to calculate eigenvalues"}

            eigenvals, eigenvecs = np.linalg.eig(A)
            return {
                "eigenvalues": eigenvals.tolist(),
                "eigenvectors": eigenvecs.tolist(),
            }
        except Exception as e:
            return {"error": f"Eigenvalue calculation error: {str(e)}"}

    def _singular_value_decomposition(
        self, matrix: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Singular value decomposition

        Args:
            matrix: Input matrix

        Returns:
            SVD decomposition result
        """
        try:
            A = np.array(matrix)
            U, s, Vt = np.linalg.svd(A)
            return {"U": U.tolist(), "singular_values": s.tolist(), "Vt": Vt.tolist()}
        except Exception as e:
            return {"error": f"SVD decomposition error: {str(e)}"}

    def _qr_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        QR decomposition

        Args:
            matrix: Input matrix

        Returns:
            QR decomposition result
        """
        try:
            A = np.array(matrix)
            Q, R = np.linalg.qr(A)
            return {"Q": Q.tolist(), "R": R.tolist()}
        except Exception as e:
            return {"error": f"QR decomposition error: {str(e)}"}

    def _lu_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        LU decomposition

        Args:
            matrix: Input matrix

        Returns:
            LU decomposition result
        """
        try:
            from scipy.linalg import lu

            A = np.array(matrix)
            P, L, U = lu(A)
            return {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}
        except Exception as e:
            return {"error": f"LU decomposition error: {str(e)}"}

    def _matrix_rank(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate matrix rank

        Args:
            matrix: Input matrix

        Returns:
            Matrix rank
        """
        try:
            A = np.array(matrix)
            rank = np.linalg.matrix_rank(A)
            return {"rank": int(rank)}
        except Exception as e:
            return {"error": f"Matrix rank calculation error: {str(e)}"}

    def _matrix_norm(
        self, matrix: List[List[float]], norm_type: Union[str, int] = "frobenius"
    ) -> Dict[str, Any]:
        """
        Calculate matrix norm

        Args:
            matrix: Input matrix
            norm_type: Norm type ('frobenius', 'nuc', 1, -1, 2, -2)

        Returns:
            Matrix norm
        """
        try:
            A = np.array(matrix)

            # Map user-friendly names to numpy parameters
            norm_map = {
                "frobenius": "fro",
                "nuc": "nuc",
                "l1": 1,
                "l2": 2,
                "inf": np.inf,
            }
            ord_param = norm_map.get(str(norm_type).lower(), norm_type)

            norm = np.linalg.norm(A, ord=ord_param)
            return {"norm": float(norm), "norm_type": str(norm_type)}
        except Exception as e:
            return {"error": f"Matrix norm calculation error: {str(e)}"}

    def _matrix_trace(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate matrix trace

        Args:
            matrix: Input matrix

        Returns:
            Matrix trace
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "Matrix must be square to calculate trace"}

            trace_val = np.trace(A)
            return {"trace": float(trace_val)}
        except Exception as e:
            return {"error": f"Matrix trace calculation error: {str(e)}"}

    def _matrix_condition_number(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate matrix condition number

        Args:
            matrix: Input matrix

        Returns:
            Matrix condition number
        """
        try:
            A = np.array(matrix)
            cond_num = np.linalg.cond(A)
            return {"condition_number": float(cond_num)}
        except Exception as e:
            return {"error": f"Condition number calculation error: {str(e)}"}

    def _matrix_power(self, matrix: List[List[float]], power: int) -> Dict[str, Any]:
        """
        Calculate matrix power

        Args:
            matrix: Input matrix
            power: Power exponent

        Returns:
            Matrix power
        """
        try:
            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {"error": "Matrix must be square to calculate power"}

            result = np.linalg.matrix_power(A, power)
            return {"result": result.tolist(), "power": power}
        except Exception as e:
            return {"error": f"Matrix power calculation error: {str(e)}"}

    def _matrix_exponential(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate matrix exponential

        Args:
            matrix: Input matrix

        Returns:
            Matrix exponential
        """
        try:
            from scipy.linalg import expm

            A = np.array(matrix)
            if A.shape[0] != A.shape[1]:
                return {
                    "error": "Matrix must be square to calculate matrix exponential"
                }

            result = expm(A)
            return {"result": result.tolist()}
        except Exception as e:
            return {"error": f"Matrix exponential calculation error: {str(e)}"}

    def _solve_linear_system(
        self, coefficients: List[List[float]], constants: List[float]
    ) -> Dict[str, Any]:
        """
        Solve linear system Ax = b

        Args:
            coefficients: Coefficient matrix A
            constants: Constant vector b

        Returns:
            Solution to the system of equations
        """
        try:
            A = np.array(coefficients)
            b = np.array(constants)

            if A.shape[0] != len(constants):
                return {
                    "error": "Number of rows in coefficient matrix must equal length of constant vector"
                }

            solution = np.linalg.solve(A, b)
            return {"solution": solution.tolist()}
        except np.linalg.LinAlgError:
            return {
                "error": "Linear system has no solution or has infinitely many solutions"
            }
        except Exception as e:
            return {"error": f"Linear system solving error: {str(e)}"}
