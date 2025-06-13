# -*- coding: utf-8 -*-
"""
Basic Numerical Computation Module
Provides basic numerical computation and mathematical operations functionality
"""

import math
from typing import List, Dict, Any, Optional, Union
import decimal
from decimal import Decimal, getcontext


class BasicCalculator:
    """Basic calculator class providing basic numerical computation and mathematical functions"""

    def __init__(self):
        """Initialize the basic calculator"""
        getcontext().prec = 50  # Set default precision

    def basic_arithmetic_tool(
        self,
        operation: str,
        numbers: List[float],
        precision: Optional[int] = None,
        use_decimal: bool = False,
    ) -> Dict[str, Any]:
        """
        Basic arithmetic operation tool

        Args:
            operation: Operation type ('add', 'subtract', 'multiply', 'divide', 'power', 'modulo', 'modulus', 'factorial', 'gcd', 'lcm', 'sum', 'product', 'average')
            numbers: List of values
            precision: Calculation precision (number of decimal places)
            use_decimal: Whether to use high-precision Decimal calculation

        Returns:
            Calculation result
        """
        try:
            if not numbers:
                return {"error": "At least one number is required"}

            if use_decimal:
                if precision:
                    getcontext().prec = (
                        precision + 10
                    )  # Extra digits to avoid rounding errors
                numbers = [Decimal(str(n)) for n in numbers]

            if operation in ["add", "sum"]:
                result = sum(numbers)
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "Addition",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation in ["multiply", "product"]:
                result = numbers[0]
                for n in numbers[1:]:
                    result *= n
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "Multiplication",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation in ["modulo", "modulus"]:
                if len(numbers) != 2:
                    return {"error": "Modulo operation requires exactly two numbers"}
                dividend, divisor = numbers
                if divisor == 0:
                    return {"error": "Divisor cannot be zero"}
                result = dividend % divisor
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "Modulo",
                    "dividend": float(dividend) if not use_decimal else str(dividend),
                    "divisor": float(divisor) if not use_decimal else str(divisor),
                }

            elif operation == "average":
                result = sum(numbers) / len(numbers)
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "Average",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation == "factorial":
                if len(numbers) != 1:
                    return {"error": "Factorial operation requires exactly one number"}
                n = numbers[0]

                if not isinstance(n, (int, float, Decimal)):
                    return {"error": "Factorial requires a numeric input"}

                is_integer = False
                if isinstance(n, int):
                    is_integer = True
                elif isinstance(n, Decimal):
                    if n == n.to_integral_value():
                        is_integer = True
                elif isinstance(n, float):
                    if n.is_integer():
                        is_integer = True

                if not is_integer:
                    return {"error": "Factorial requires integer input"}

                n_int = int(n)
                if n_int < 0:
                    return {"error": "Factorial requires a non-negative integer"}

                # Limit to avoid overflow
                if n_int > 1000:
                    return {
                        "error": "Number too large, factorial is limited to 1000 or below"
                    }

                result = math.factorial(n_int)
                return {
                    "result": result,
                    "operation": "Factorial",
                    "input": n_int,
                }

            elif operation == "gcd":
                if len(numbers) < 2:
                    return {"error": "GCD requires at least two numbers"}
                integers = [
                    int(n)
                    for n in numbers
                    if isinstance(n, (int, float, Decimal)) and n == int(n)
                ]
                if len(integers) != len(numbers):
                    return {"error": "GCD requires integer inputs"}
                result = integers[0]
                for n in integers[1:]:
                    result = math.gcd(result, n)
                return {
                    "result": result,
                    "operation": "GCD",
                    "inputs": integers,
                }

            elif operation == "lcm":
                if len(numbers) < 2:
                    return {"error": "LCM requires at least two numbers"}
                integers = [
                    int(n)
                    for n in numbers
                    if isinstance(n, (int, float, Decimal)) and n == int(n)
                ]
                if len(integers) != len(numbers):
                    return {"error": "LCM requires integer inputs"}
                result = integers[0]
                for n in integers[1:]:
                    result = abs(result * n) // math.gcd(result, n)
                return {
                    "result": result,
                    "operation": "LCM",
                    "inputs": integers,
                }

            else:
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Arithmetic error: {str(e)}"}

    def mathematical_functions_tool(
        self,
        function: str,
        value: float,
        base: Optional[float] = None,
        precision: Optional[int] = None,
        angle_unit: str = "radians",
    ) -> Dict[str, Any]:
        """
        Mathematical function calculation tool

        Args:
            function: Function type ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                                    'log', 'log10', 'ln', 'sqrt', 'cbrt', 'exp', 'abs', 'ceil', 'floor', 'round', 'factorial', 'gamma')
            value: Input value
            base: Base for logarithm (optional)
            precision: Result precision
            angle_unit: Angle unit ('radians', 'degrees')

        Returns:
            Function computation result
        """
        try:
            # Angle conversion
            if function in ["sin", "cos", "tan"] and angle_unit == "degrees":
                value = math.radians(value)

            # Basic trigonometric functions
            if function == "sin":
                result = math.sin(value)
                unit_note = (
                    f"(Input angle unit: {angle_unit})"
                    if angle_unit == "degrees"
                    else ""
                )
            elif function == "cos":
                result = math.cos(value)
                unit_note = (
                    f"(Input angle unit: {angle_unit})"
                    if angle_unit == "degrees"
                    else ""
                )
            elif function == "tan":
                result = math.tan(value)
                unit_note = (
                    f"(Input angle unit: {angle_unit})"
                    if angle_unit == "degrees"
                    else ""
                )

            # Inverse trigonometric functions
            elif function == "asin":
                if not -1 <= value <= 1:
                    return {"error": "Input for asin must be in [-1, 1]"}
                result = math.asin(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"(Output angle unit: {angle_unit})"
            elif function == "acos":
                if not -1 <= value <= 1:
                    return {"error": "Input for acos must be in [-1, 1]"}
                result = math.acos(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"(Output angle unit: {angle_unit})"
            elif function == "atan":
                result = math.atan(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"(Output angle unit: {angle_unit})"

            # Hyperbolic functions
            elif function == "sinh":
                result = math.sinh(value)
                unit_note = ""
            elif function == "cosh":
                result = math.cosh(value)
                unit_note = ""
            elif function == "tanh":
                result = math.tanh(value)
                unit_note = ""

            # Logarithmic functions
            elif function == "log":
                if value <= 0:
                    return {"error": "Input for logarithm must be greater than 0"}
                if base is None:
                    base = 10
                result = math.log(value, base)
                unit_note = f"(Base: {base})"
            elif function == "log10":
                if value <= 0:
                    return {"error": "Input for log10 must be greater than 0"}
                result = math.log10(value)
                unit_note = "(Common logarithm, base: 10)"
            elif function == "ln":
                if value <= 0:
                    return {
                        "error": "Input for natural logarithm must be greater than 0"
                    }
                result = math.log(value)
                unit_note = "(Natural logarithm, base: e)"

            # Root and exponentials
            elif function == "sqrt":
                if value < 0:
                    return {"error": "Input for square root cannot be negative"}
                result = math.sqrt(value)
                unit_note = ""
            elif function == "cbrt":
                result = math.pow(value, 1 / 3)
                unit_note = ""
            elif function == "exp":
                result = math.exp(value)
                unit_note = ""

            # Other mathematical functions
            elif function == "abs":
                result = abs(value)
                unit_note = ""
            elif function == "ceil":
                result = math.ceil(value)
                unit_note = ""
            elif function == "floor":
                result = math.floor(value)
                unit_note = ""
            elif function == "round":
                if precision is not None:
                    result = round(value, precision)
                else:
                    result = round(value)
                unit_note = (
                    f"(Precision: {precision} decimal places)"
                    if precision is not None
                    else ""
                )

            # Added factorial function
            elif function == "factorial":
                if value < 0 or int(value) != value:
                    return {
                        "error": "Factorial function requires non-negative integer input"
                    }
                result = math.factorial(int(value))
                unit_note = ""

            # Added gamma function
            elif function == "gamma":
                if value <= 0:
                    return {"error": "Gamma function input must be greater than 0"}
                result = math.gamma(value)
                unit_note = ""

            else:
                return {"error": f"Unsupported mathematical function: {function}"}

            # Format result
            if precision is not None and function not in ["ceil", "floor", "round"]:
                result = round(result, precision)

            return {
                "result": result,
                "function": function,
                "input": value,
                "note": unit_note,
            }

        except Exception as e:
            return {"error": f"Mathematical function error: {str(e)}"}

    def number_converter_tool(
        self,
        number: Union[str, float, int],
        from_base: int = 10,
        to_base: int = 10,
        operation: str = "convert",
        precision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Number base conversion and formatting tool

        Args:
            number: Input number
            from_base: Source base (2-36)
            to_base: Target base (2-36)
            operation: Operation type ('convert', 'format', 'scientific', 'fraction')
            precision: Precision control

        Returns:
            Conversion result
        """
        try:
            if operation == "convert":
                # Base conversion
                if from_base < 2 or from_base > 36 or to_base < 2 or to_base > 36:
                    return {"error": "Base must be in the range 2-36"}

                # Convert to decimal
                if from_base != 10:
                    try:
                        decimal_value = int(str(number), from_base)
                    except ValueError:
                        return {
                            "error": f"Failed to convert '{number}' from base {from_base} to decimal"
                        }
                else:
                    decimal_value = int(float(number))

                # Convert from decimal to target base
                if to_base == 10:
                    result = str(decimal_value)
                else:
                    if decimal_value == 0:
                        result = "0"
                    else:
                        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        result = ""
                        value = abs(decimal_value)
                        while value > 0:
                            result = digits[value % to_base] + result
                            value //= to_base
                        if decimal_value < 0:
                            result = "-" + result

                return {
                    "result": result,
                    "original": str(number),
                    "from_base": from_base,
                    "to_base": to_base,
                    "decimal_value": decimal_value,
                }

            elif operation == "format":
                # Number formatting
                num_value = float(number)
                formats = {
                    "standard": f"{num_value:,}",
                    "scientific": f"{num_value:.6e}",
                    "percentage": f"{num_value:.2%}",
                    "currency": f"¥{num_value:,.2f}",
                }
                if precision is not None:
                    formats["fixed"] = f"{num_value:.{precision}f}"

                return {
                    "result": formats,
                    "original": number,
                    "operation": "format",
                }

            elif operation == "scientific":
                # Scientific notation
                num_value = float(number)
                mantissa = num_value
                exponent = 0

                if num_value != 0:
                    while abs(mantissa) >= 10:
                        mantissa /= 10
                        exponent += 1
                    while abs(mantissa) < 1:
                        mantissa *= 10
                        exponent -= 1

                if precision is not None:
                    mantissa = round(mantissa, precision)

                return {
                    "result": f"{mantissa} × 10^{exponent}",
                    "mantissa": mantissa,
                    "exponent": exponent,
                    "standard_notation": f"{mantissa}e{exponent}",
                    "original": number,
                }

            elif operation == "fraction":
                # Convert to fraction
                from fractions import Fraction

                try:
                    frac = Fraction(float(number)).limit_denominator(10000)
                    return {
                        "result": f"{frac.numerator}/{frac.denominator}",
                        "numerator": frac.numerator,
                        "denominator": frac.denominator,
                        "decimal": float(frac),
                        "original": number,
                    }
                except Exception:
                    return {"error": "Unable to convert to fractional form"}

            else:
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Number conversion error: {str(e)}"}

    def unit_converter_tool(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        unit_type: str,
    ) -> Dict[str, Any]:
        """
        Unit conversion tool

        Args:
            value: Input value
            from_unit: Source unit
            to_unit: Target unit
            unit_type: Unit type ('length', 'weight', 'temperature', 'area', 'volume', 'time', 'speed', 'energy')

        Returns:
            Conversion result
        """
        try:
            # Conversion factors (all converted to base unit)
            conversions = {
                "length": {  # Base unit: meter (m)
                    "mm": 0.001,
                    "cm": 0.01,
                    "m": 1,
                    "km": 1000,
                    "in": 0.0254,
                    "ft": 0.3048,
                    "yd": 0.9144,
                    "mi": 1609.34,
                },
                "weight": {  # Base unit: kilogram (kg)
                    "mg": 0.000001,
                    "g": 0.001,
                    "kg": 1,
                    "t": 1000,
                    "oz": 0.0283495,
                    "lb": 0.453592,
                    "st": 6.35029,
                },
                "temperature": {  # Special handling
                    "celsius": "celsius",
                    "fahrenheit": "fahrenheit",
                    "kelvin": "kelvin",
                },
                "area": {  # Base unit: square meter (m²)
                    "mm2": 0.000001,
                    "cm2": 0.0001,
                    "m2": 1,
                    "km2": 1000000,
                    "in2": 0.00064516,
                    "ft2": 0.092903,
                    "yd2": 0.836127,
                },
                "volume": {  # Base unit: liter (L)
                    "ml": 0.001,
                    "l": 1,
                    "m3": 1000,
                    "tsp": 0.00492892,
                    "tbsp": 0.0147868,
                    "cup": 0.236588,
                    "pt": 0.473176,
                    "qt": 0.946353,
                    "gal": 3.78541,
                },
                "time": {  # Base unit: second (s)
                    "ms": 0.001,
                    "s": 1,
                    "min": 60,
                    "h": 3600,
                    "day": 86400,
                    "week": 604800,
                    "month": 2629746,
                    "year": 31556952,
                },
                "speed": {  # Base unit: meter/second (m/s)
                    "m/s": 1,
                    "km/h": 0.277778,
                    "mph": 0.44704,
                    "knot": 0.514444,
                },
                "energy": {  # Base unit: joule (J)
                    "j": 1,
                    "kj": 1000,
                    "cal": 4.184,
                    "kcal": 4184,
                    "kwh": 3600000,
                },
            }

            if unit_type not in conversions:
                return {"error": f"Unsupported unit type: {unit_type}"}

            unit_map = conversions[unit_type]

            # Special case for temperature conversion
            if unit_type == "temperature":
                result = self._convert_temperature(value, from_unit, to_unit)
                if isinstance(result, dict) and "error" in result:
                    return result
            else:
                if from_unit not in unit_map or to_unit not in unit_map:
                    return {"error": f"Unsupported unit: {from_unit} or {to_unit}"}

                # Convert to base unit, then to target unit
                base_value = value * unit_map[from_unit]
                result = base_value / unit_map[to_unit]

            return {
                "result": result,
                "original_value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "unit_type": unit_type,
                "formatted": f"{value} {from_unit} = {result} {to_unit}",
            }

        except Exception as e:
            return {"error": f"Unit conversion error: {str(e)}"}

    def _convert_temperature(
        self, value: float, from_unit: str, to_unit: str
    ) -> Union[float, Dict[str, str]]:
        """Helper for temperature conversion"""
        try:
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()

            # Convert to Celsius first
            if from_unit == "celsius":
                celsius = value
            elif from_unit == "fahrenheit":
                celsius = (value - 32) * 5 / 9
            elif from_unit == "kelvin":
                celsius = value - 273.15
            else:
                return {"error": f"Unsupported temperature unit: {from_unit}"}

            # Convert from Celsius to target unit
            if to_unit == "celsius":
                result = celsius
            elif to_unit == "fahrenheit":
                result = celsius * 9 / 5 + 32
            elif to_unit == "kelvin":
                result = celsius + 273.15
            else:
                return {"error": f"Unsupported temperature unit: {to_unit}"}

            return result

        except Exception as e:
            return {"error": f"Temperature conversion error: {str(e)}"}

    def precision_calculator_tool(
        self,
        numbers: List[float],
        operation: str,
        precision_digits: int = 10,
        rounding_mode: str = "round_half_up",
    ) -> Dict[str, Any]:
        """
        High-precision calculation tool

        Args:
            numbers: List of numbers
            operation: Operation type ('add', 'subtract', 'multiply', 'divide', 'power', 'sqrt')
            precision_digits: Precision digits
            rounding_mode: Rounding mode

        Returns:
            High-precision calculation result
        """
        try:
            # Set precision and rounding mode
            getcontext().prec = precision_digits + 10

            rounding_modes = {
                "round_half_up": decimal.ROUND_HALF_UP,
                "round_half_down": decimal.ROUND_HALF_DOWN,
                "round_half_even": decimal.ROUND_HALF_EVEN,
                "round_up": decimal.ROUND_UP,
                "round_down": decimal.ROUND_DOWN,
                "round_ceiling": decimal.ROUND_CEILING,
                "round_floor": decimal.ROUND_FLOOR,
            }

            if rounding_mode in rounding_modes:
                getcontext().rounding = rounding_modes[rounding_mode]

            decimal_numbers = [Decimal(str(n)) for n in numbers]

            if operation == "add":
                result = sum(decimal_numbers)
            elif operation == "subtract":
                if len(decimal_numbers) < 2:
                    return {"error": "Subtraction requires at least two numbers"}
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    result -= n
            elif operation == "multiply":
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    result *= n
            elif operation == "divide":
                if len(decimal_numbers) < 2:
                    return {"error": "Division requires at least two numbers"}
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    if n == 0:
                        return {"error": "Divisor cannot be zero"}
                    result /= n
            elif operation == "power":
                if len(decimal_numbers) != 2:
                    return {"error": "Power operation requires exactly two numbers"}
                result = decimal_numbers[0] ** decimal_numbers[1]
            elif operation == "sqrt":
                if len(decimal_numbers) != 1:
                    return {
                        "error": "Square root operation requires exactly one number"
                    }
                if decimal_numbers[0] < 0:
                    return {"error": "Input for square root cannot be negative"}
                result = decimal_numbers[0].sqrt()
            else:
                return {"error": f"Unsupported operation type: {operation}"}

            quantized_result = result.quantize(Decimal("0." + "0" * precision_digits))

            return {
                "result": str(quantized_result),
                "result_float": float(quantized_result),
                "operation": operation,
                "precision_digits": precision_digits,
                "rounding_mode": rounding_mode,
                "inputs": [str(n) for n in decimal_numbers],
            }

        except Exception as e:
            return {"error": f"High-precision calculation error: {str(e)}"}

    def number_properties_tool(
        self,
        number: Union[int, float],
        analysis_type: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Number properties analysis tool

        Args:
            number: Input number
            analysis_type: Analysis type ('comprehensive', 'prime', 'divisors', 'properties')

        Returns:
            Analysis results of number properties
        """
        try:
            if isinstance(number, float) and not number.is_integer():
                return {"error": "Some analytical features only support integers"}

            n = (
                int(number)
                if isinstance(number, float) and number.is_integer()
                else number
            )
            result = {"number": n}

            if analysis_type in ["comprehensive", "properties"]:
                # Basic properties
                result.update(
                    {
                        "is_integer": isinstance(n, int),
                        "is_positive": n > 0,
                        "is_negative": n < 0,
                        "is_zero": n == 0,
                        "absolute_value": abs(n),
                    }
                )

                if isinstance(n, int):
                    result.update(
                        {
                            "is_even": n % 2 == 0,
                            "is_odd": n % 2 == 1,
                            "is_perfect_square": (
                                int(n**0.5) ** 2 == n if n >= 0 else False
                            ),
                            "is_perfect_cube": (
                                round(n ** (1 / 3)) ** 3 == n if n >= 0 else False
                            ),
                        }
                    )

            if (
                analysis_type in ["comprehensive", "prime"]
                and isinstance(n, int)
                and n > 1
            ):
                # Prime check
                result["is_prime"] = self._is_prime(n)

                # Prime factorization
                if n <= 10000:
                    result["prime_factors"] = self._prime_factorization(n)

            if (
                analysis_type in ["comprehensive", "divisors"]
                and isinstance(n, int)
                and 0 < n <= 1000
            ):
                # Divisors analysis
                divisors = self._get_divisors(n)
                result.update(
                    {
                        "divisors": divisors,
                        "divisor_count": len(divisors),
                        "sum_of_divisors": sum(divisors),
                        "is_perfect_number": sum(divisors[:-1]) == n,
                    }
                )

            if analysis_type == "comprehensive" and isinstance(n, int):
                # Digital root
                result["digital_root"] = self._digital_root(abs(n))
                # Digit sum
                result["digit_sum"] = sum(int(digit) for digit in str(abs(n)))

            return result

        except Exception as e:
            return {"error": f"Number properties analysis error: {str(e)}"}

    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _prime_factorization(self, n: int) -> List[int]:
        """Prime factorization"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def _get_divisors(self, n: int) -> List[int]:
        """Get all divisors"""
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def _digital_root(self, n: int) -> int:
        """Calculate digital root"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n
