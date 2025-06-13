# -*- coding: utf-8 -*-
"""
Complex Analysis Calculation Module
Provides comprehensive and rich complex function analysis capabilities
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
    """Complex Analysis Calculator class, providing comprehensive complex function analysis capabilities"""

    def __init__(self):
        """Initialize complex analysis calculator"""
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
        Comprehensive complex analysis calculation tool

        Args:
            operation: Operation type
            complex_number: Single complex number
            complex_numbers: List of complex numbers
            function_expression: Complex function expression
            variable: Complex variable name
            contour_points: Integration contour points
            singularities: List of singularities
            center: Expansion center
            radius: Convergence radius
            order: Pole order or number of series terms
            x_range: Real axis range
            y_range: Imaginary axis range
            resolution: Plotting resolution
            colormap: Color mapping
            filename: Save filename
            plot_type: Plot type
            series_terms: Number of series terms
            branch_cut: Branch cut
            method: Calculation method

        Returns:
            Complex analysis calculation results
        """
        try:
            if operation == "convert_form":
                if complex_number is None:
                    return {"error": "Conversion operation requires a complex number"}
                return self._convert_complex_form(complex_number)

            elif operation == "arithmetic":
                if complex_numbers is None or len(complex_numbers) < 2:
                    return {
                        "error": "Complex arithmetic requires at least two complex numbers"
                    }
                return self._complex_arithmetic(complex_numbers, method)

            elif operation == "function_evaluation":
                if function_expression is None or complex_number is None:
                    return {
                        "error": "Function evaluation requires function expression and complex number"
                    }
                return self._evaluate_complex_function(
                    function_expression, complex_number, variable
                )

            elif operation == "residue_calculation":
                if function_expression is None or singularities is None:
                    return {
                        "error": "Residue calculation requires function expression and singularities"
                    }
                return self._calculate_residues(
                    function_expression, singularities, variable, order
                )

            elif operation == "contour_integration":
                if function_expression is None or contour_points is None:
                    return {
                        "error": "Contour integration requires function expression and integration path"
                    }
                return self._contour_integration(
                    function_expression, contour_points, variable
                )

            elif operation == "series_expansion":
                if function_expression is None:
                    return {"error": "Series expansion requires function expression"}
                return self._series_expansion(
                    function_expression, variable, center, series_terms
                )

            elif operation == "analytic_continuation":
                if function_expression is None:
                    return {
                        "error": "Analytic continuation requires function expression"
                    }
                return self._analytic_continuation(
                    function_expression, variable, center, radius, branch_cut
                )

            elif operation == "complex_plot":
                if function_expression is None:
                    return {
                        "error": "Complex function plotting requires function expression"
                    }
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
                    return {"error": "Conformal mapping requires function expression"}
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
                    return {
                        "error": "Singularity analysis requires function expression"
                    }
                return self._analyze_singularities(
                    function_expression, variable, x_range, y_range
                )

            else:
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Complex analysis calculation error: {str(e)}"}

    def _convert_complex_form(self, z: Union[str, complex]) -> Dict[str, Any]:
        """
        Complex number form conversion

        Args:
            z: Complex number

        Returns:
            Various forms of complex number representation
        """
        try:
            # Parse complex number
            if isinstance(z, str):
                # Handle string form of complex number
                z_parsed = complex(z.replace("i", "j").replace("I", "j"))
            else:
                z_parsed = complex(z)

            # Extract real and imaginary parts
            real_part = z_parsed.real
            imag_part = z_parsed.imag

            # Calculate modulus and argument
            modulus = abs(z_parsed)
            argument = cmath.phase(z_parsed)
            argument_degrees = np.degrees(argument)

            # Exponential form
            exponential_form = f"{modulus:.6f} * exp({argument:.6f}j)"

            # Trigonometric form
            trigonometric_form = (
                f"{modulus:.6f} * (cos({argument:.6f}) + j*sin({argument:.6f}))"
            )

            # Polar form
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
            return {"error": f"Complex number conversion error: {str(e)}"}

    def _complex_arithmetic(
        self, numbers: List[Union[str, complex]], operation: str = "all"
    ) -> Dict[str, Any]:
        """
        Complex arithmetic operations

        Args:
            numbers: List of complex numbers
            operation: Operation type

        Returns:
            Operation results
        """
        try:
            # Parse complex numbers
            parsed_numbers = []
            for num in numbers:
                if isinstance(num, str):
                    parsed_numbers.append(
                        complex(num.replace("i", "j").replace("I", "j"))
                    )
                else:
                    parsed_numbers.append(complex(num))

            if len(parsed_numbers) < 2:
                return {"error": "At least two complex numbers required for operation"}

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
                    results["division"] = {"error": "Division by zero error"}

            if operation in ["all", "power"]:
                results["power"] = {"z1^z2": z1**z2, "z2^z1": z2**z1}

            # If there are more complex numbers, perform chain operations
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
            return {"error": f"Complex arithmetic error: {str(e)}"}

    def _evaluate_complex_function(
        self, func_expr: str, z_value: Union[str, complex], variable: str = "z"
    ) -> Dict[str, Any]:
        """
        Complex function evaluation

        Args:
            func_expr: Function expression
            z_value: Complex value
            variable: Variable name

        Returns:
            Function value and related information
        """
        try:
            # Parse complex number
            if isinstance(z_value, str):
                z_parsed = complex(z_value.replace("i", "j").replace("I", "j"))
            else:
                z_parsed = complex(z_value)

            # Use sympy for symbolic computation
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # Calculate function value
            func_value = complex(expr.subs(z_sym, z_parsed))

            # Calculate derivative
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
            return {"error": f"Complex function evaluation error: {str(e)}"}

    def _calculate_residues(
        self,
        func_expr: str,
        singularities: List[Union[str, complex]],
        variable: str = "z",
        order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate residues

        Args:
            func_expr: Function expression
            singularities: List of singularities
            variable: Variable name
            order: Pole order

        Returns:
            Residue calculation results
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            residues_results = {}
            total_residue = 0

            for i, singularity in enumerate(singularities):
                # Parse singularity
                if isinstance(singularity, str):
                    sing_point = complex(
                        singularity.replace("i", "j").replace("I", "j")
                    )
                else:
                    sing_point = complex(singularity)

                try:
                    # Calculate residue
                    residue = sp.residue(expr, z_sym, sing_point)
                    residue_value = complex(residue)
                    total_residue += residue_value

                    # Analyze singularity type
                    # Calculate Laurent series expansion
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
                        "error": f"Cannot calculate residue: {str(e)}",
                    }

            return {
                "function": func_expr,
                "singularities": residues_results,
                "total_residue": total_residue,
                "residue_theorem": f"Contour integral value = 2πi × {total_residue} = {2j * np.pi * total_residue}",
            }

        except Exception as e:
            return {"error": f"Residue calculation error: {str(e)}"}

    def _classify_singularity(self, laurent_series, point) -> str:
        """Classify singularity type"""
        try:
            # Simplified singularity classification
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
        Contour integration

        Args:
            func_expr: Function expression
            contour_points: Integration contour points
            variable: Variable name

        Returns:
            Integration results
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # Convert path points to complex numbers
            contour_complex = [complex(point[0], point[1]) for point in contour_points]

            # Numerical integration
            def integrand(t):
                # Parameterize path
                n = len(contour_complex)
                if n < 2:
                    return 0

                # Linear interpolation path
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

                # Calculate integrand value
                try:
                    f_val = complex(expr.subs(z_sym, z_t))
                    return f_val * dz_dt * (n - 1)  # Multiply by path length factor
                except:
                    return 0

            # Numerical integration
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
            return {"error": f"Contour integration calculation error: {str(e)}"}

    def _series_expansion(
        self,
        func_expr: str,
        variable: str = "z",
        center: Optional[Union[str, complex]] = None,
        terms: int = 10,
    ) -> Dict[str, Any]:
        """
        Series expansion

        Args:
            func_expr: Function expression
            variable: Variable name
            center: Expansion center
            terms: Number of series terms

        Returns:
            Series expansion results
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # Parse expansion center
            if center is None:
                center_point = 0
            elif isinstance(center, str):
                center_point = complex(center.replace("i", "j").replace("I", "j"))
            else:
                center_point = complex(center)

            # Taylor series expansion
            try:
                taylor_series = sp.series(expr, z_sym, center_point, n=terms)
                taylor_coefficients = []

                # Extract coefficients
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
                taylor_series = f"Cannot expand: {str(e)}"
                taylor_coefficients = []

            # Laurent series expansion (if possible)
            try:
                laurent_series = sp.series(expr, z_sym, center_point, n=terms)
                laurent_str = str(laurent_series)
            except:
                laurent_series = "Cannot perform Laurent expansion"
                laurent_str = ""

            # Convergence radius estimation
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
                        else "unknown"
                    ),
                },
            }

        except Exception as e:
            return {"error": f"Series expansion error: {str(e)}"}

    def _estimate_convergence_radius(
        self, coefficients: List[complex]
    ) -> Optional[float]:
        """Estimate convergence radius"""
        try:
            if len(coefficients) < 2:
                return None

            # Use ratio test
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
        Analytic continuation

        Args:
            func_expr: Function expression
            variable: Variable name
            center: Continuation center
            radius: Continuation radius
            branch_cut: Branch cut

        Returns:
            Analytic continuation results
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # Parse center point
            if center is None:
                center_point = 0
            elif isinstance(center, str):
                center_point = complex(center.replace("i", "j").replace("I", "j"))
            else:
                center_point = complex(center)

            # Analyze function singularities
            singularities = []
            try:
                # Find points where denominator is zero
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

            # Calculate continuation domain
            if radius is None:
                if singularities:
                    # Find nearest singularity
                    distances = [
                        abs(complex(s) - center_point)
                        for s in singularities
                        if isinstance(s, (int, float, complex))
                    ]
                    if distances:
                        radius = (
                            min(distances) * 0.9
                        )  # Slightly less than nearest singularity distance
                    else:
                        radius = 1.0
                else:
                    radius = float("inf")

            # Branch cut analysis
            branch_info = {}
            if "log" in func_expr or "sqrt" in func_expr or "**" in func_expr:
                branch_info = {
                    "has_branch_cuts": True,
                    "suggested_cuts": self._analyze_branch_cuts(func_expr),
                    "principal_branch": "Using principal branch",
                }
            else:
                branch_info = {
                    "has_branch_cuts": False,
                    "note": "Function is single-valued on complex plane",
                }

            return {
                "function": func_expr,
                "continuation_center": str(center_point),
                "continuation_radius": radius,
                "domain_of_analyticity": (
                    f"|z - {center_point}| < {radius}"
                    if radius != float("inf")
                    else "Entire complex plane"
                ),
                "singularities": [str(s) for s in singularities],
                "branch_analysis": branch_info,
                "continuation_method": (
                    "Power series expansion"
                    if radius < float("inf")
                    else "Entire function"
                ),
            }

        except Exception as e:
            return {"error": f"Analytic continuation analysis error: {str(e)}"}

    def _analyze_branch_cuts(self, func_expr: str) -> List[str]:
        """Analyze branch cuts"""
        cuts = []
        if "log" in func_expr:
            cuts.append("Negative real axis: arg(z) = π")
        if "sqrt" in func_expr:
            cuts.append("Negative real axis: arg(z) = π")
        if "**" in func_expr and ("1/" in func_expr or "0.5" in func_expr):
            cuts.append("Possible branch cut along negative real axis")
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
        Complex function visualization

        Args:
            func_expr: Function expression
            variable: Variable name
            x_range: Real axis range
            y_range: Imaginary axis range
            resolution: Resolution
            plot_type: Plot type
            colormap: Color mapping
            filename: Filename

        Returns:
            Plotting results
        """
        try:
            # Create complex plane grid
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y

            # Calculate function values
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)
            func_lambdified = sp.lambdify(z_sym, expr, "numpy")

            try:
                W = func_lambdified(Z)
            except:
                # Point-wise calculation
                W = np.zeros_like(Z, dtype=complex)
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        try:
                            W[i, j] = complex(expr.subs(z_sym, Z[i, j]))
                        except:
                            W[i, j] = np.nan + 1j * np.nan

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Complex Function Analysis: f(z) = {func_expr}", fontsize=14)

            # 1. Domain coloring plot
            if plot_type in ["domain_coloring", "all"]:
                ax1 = axes[0, 0]

                # Calculate colors
                magnitude = np.abs(W)
                phase = np.angle(W)

                # Normalization
                magnitude_norm = np.log(1 + magnitude) / np.log(
                    1 + np.nanmax(magnitude)
                )
                phase_norm = (phase + np.pi) / (2 * np.pi)

                # HSV color space
                H = phase_norm
                S = np.ones_like(magnitude_norm)
                V = magnitude_norm

                # Convert to RGB
                HSV = np.stack([H, S, V], axis=-1)
                RGB = hsv_to_rgb(HSV)

                ax1.imshow(
                    RGB,
                    extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                    origin="lower",
                    interpolation="bilinear",
                )
                ax1.set_title("Domain Coloring")
                ax1.set_xlabel("Re(z)")
                ax1.set_ylabel("Im(z)")
                ax1.grid(True, alpha=0.3)

            # 2. Magnitude plot
            ax2 = axes[0, 1]
            magnitude_plot = ax2.contourf(X, Y, np.abs(W), levels=20, cmap="viridis")
            ax2.set_title("Magnitude |f(z)|")
            ax2.set_xlabel("Re(z)")
            ax2.set_ylabel("Im(z)")
            plt.colorbar(magnitude_plot, ax=ax2)

            # 3. Phase plot
            ax3 = axes[1, 0]
            phase_plot = ax3.contourf(X, Y, np.angle(W), levels=20, cmap="hsv")
            ax3.set_title("Phase arg(f(z))")
            ax3.set_xlabel("Re(z)")
            ax3.set_ylabel("Im(z)")
            plt.colorbar(phase_plot, ax=ax3)

            # 4. Real and imaginary parts
            ax4 = axes[1, 1]
            real_plot = ax4.contour(
                X, Y, np.real(W), levels=10, colors="red", alpha=0.7
            )
            imag_plot = ax4.contour(
                X, Y, np.imag(W), levels=10, colors="blue", alpha=0.7
            )
            ax4.set_title("Real Part (Red) and Imaginary Part (Blue)")
            ax4.set_xlabel("Re(z)")
            ax4.set_ylabel("Im(z)")
            ax4.legend(["Re(f(z))", "Im(f(z))"])

            plt.tight_layout()

            # Save image
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
            return {"error": f"Complex function plotting error: {str(e)}"}

    def _detect_symbolic_singularities(self, expr, variable, x_range, y_range):
        """Use symbolic method to detect singularities within the plotting region"""
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
                    # Non-numeric roots
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
        Conformal mapping visualization

        Args:
            func_expr: Mapping function expression
            variable: Variable name
            x_range: Real axis range
            y_range: Imaginary axis range
            resolution: Grid resolution
            filename: Filename

        Returns:
            Conformal mapping results
        """
        try:
            # Create grid
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)

            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)
            func_lambdified = sp.lambdify(z_sym, expr, "numpy")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Original domain
            ax1.set_title("Original Domain (z-plane)")
            ax1.set_xlabel("Re(z)")
            ax1.set_ylabel("Im(z)")
            ax1.grid(True)
            ax1.set_aspect("equal")

            # Draw grid lines
            for xi in x[::2]:  # Vertical lines
                z_line = xi + 1j * y
                ax1.plot([xi] * len(y), y, "b-", alpha=0.5)

            for yi in y[::2]:  # Horizontal lines
                z_line = x + 1j * yi
                ax1.plot(x, [yi] * len(x), "r-", alpha=0.5)

            # Mapped domain
            ax2.set_title(f"Mapped Domain w = f(z) = {func_expr}")
            ax2.set_xlabel("Re(w)")
            ax2.set_ylabel("Im(w)")
            ax2.grid(True)
            ax2.set_aspect("equal")

            # Map grid lines
            for xi in x[::2]:  # Map vertical lines
                z_line = xi + 1j * y
                try:
                    w_line = func_lambdified(z_line)
                    ax2.plot(np.real(w_line), np.imag(w_line), "b-", alpha=0.7)
                except:
                    pass

            for yi in y[::2]:  # Map horizontal lines
                z_line = x + 1j * yi
                try:
                    w_line = func_lambdified(z_line)
                    ax2.plot(np.real(w_line), np.imag(w_line), "r-", alpha=0.7)
                except:
                    pass

            plt.tight_layout()

            # Save image
            if filename is None:
                filename = "conformal_mapping"

            filepath, _ = generate_unique_filename("conformal_mapping", "png", filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            # Analyze conformal properties
            jacobian_analysis = self._analyze_jacobian(func_expr, variable)

            return {
                "mapping_function": func_expr,
                "domain": f"[{x_range[0]}, {x_range[1]}] + i[{y_range[0]}, {y_range[1]}]",
                "file_path": filepath,
                "conformal_analysis": jacobian_analysis,
                "grid_resolution": f"{resolution}x{resolution}",
            }

        except Exception as e:
            return {"error": f"Conformal mapping visualization error: {str(e)}"}

    def _analyze_jacobian(self, func_expr: str, variable: str) -> Dict[str, Any]:
        """Analyze Jacobian matrix and conformal properties"""
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            # Calculate derivative
            derivative = sp.diff(expr, z_sym)

            return {
                "derivative": str(derivative),
                "conformal_condition": "f'(z) ≠ 0",
                "note": "The mapping is conformal at points where f'(z) ≠ 0",
            }
        except:
            return {"note": "Cannot analyze conformal properties"}

    def _analyze_singularities(
        self,
        func_expr: str,
        variable: str = "z",
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
    ) -> Dict[str, Any]:
        """
        Singularity analysis

        Args:
            func_expr: Function expression
            variable: Variable name
            x_range: Real axis range
            y_range: Imaginary axis range

        Returns:
            Singularity analysis results
        """
        try:
            z_sym = sp.Symbol(variable)
            expr = sp.sympify(func_expr)

            singularities = []

            # Find poles (points where denominator is zero)
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

                                # Analyze pole order
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
                                    "note": "Symbolic form pole",
                                }
                            )
            except:
                pass

            # Find branch points
            if any(func in func_expr for func in ["sqrt", "log", "**"]):
                branch_points = self._find_branch_points(expr, z_sym)
                singularities.extend(branch_points)

            # Find essential singularities
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
            return {"error": f"Singularity analysis error: {str(e)}"}

    def _find_pole_order(self, expr, variable, pole_point) -> int:
        """Determine pole order"""
        try:
            # Calculate Laurent series
            series = sp.series(expr, variable, pole_point, n=10)
            series_str = str(series)

            # Find highest negative power
            import re

            negative_powers = re.findall(r"\*\*\(-(\d+)\)", series_str)
            if negative_powers:
                return max(int(power) for power in negative_powers)
            return 1
        except:
            return 1

    def _find_branch_points(self, expr, variable) -> List[Dict[str, Any]]:
        """Find branch points"""
        branch_points = []
        try:
            # Simplified branch point detection
            expr_str = str(expr)
            if "log" in expr_str:
                branch_points.append(
                    {
                        "point": "0",
                        "type": "branch_point",
                        "function": "logarithm",
                        "note": "Branch point of logarithm function",
                    }
                )
            if "sqrt" in expr_str:
                branch_points.append(
                    {
                        "point": "0",
                        "type": "branch_point",
                        "function": "square_root",
                        "note": "Branch point of square root function",
                    }
                )
        except:
            pass
        return branch_points

    def _find_essential_singularities(self, expr, variable) -> List[Dict[str, Any]]:
        """Find essential singularities"""
        essential = []
        try:
            expr_str = str(expr)
            if "exp" in expr_str and ("1/" in expr_str or "**(-" in expr_str):
                essential.append(
                    {
                        "type": "essential_singularity",
                        "note": "May contain essential singularities, requires further analysis",
                    }
                )
        except:
            pass
        return essential

    def _classify_singularities(
        self, singularities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Classify singularity statistics"""
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
