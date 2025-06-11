# -*- coding: utf-8 -*-
"""
基础数值计算模块
提供基础的数值计算和数学运算功能
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Union
import decimal
from decimal import Decimal, getcontext
import warnings
import os

# 设置环境变量抑制NumPy警告
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')


class BasicCalculator:
    """基础计算器类，提供基础的数值计算和数学运算功能"""

    def __init__(self):
        """初始化基础计算器"""
        # 设置默认精度
        getcontext().prec = 50

    def basic_arithmetic_tool(
        self,
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
            if not numbers:
                return {"error": "需要至少一个数值"}

            if use_decimal:
                if precision:
                    getcontext().prec = precision + 10  # 额外精度避免舍入误差
                numbers = [Decimal(str(n)) for n in numbers]

            if operation == "add":
                result = sum(numbers)
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "加法",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation == "subtract":
                if len(numbers) < 2:
                    return {"error": "减法需要至少两个数值"}
                result = numbers[0]
                for n in numbers[1:]:
                    result -= n
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "减法",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation == "multiply":
                result = numbers[0]
                for n in numbers[1:]:
                    result *= n
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "乘法",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation == "divide":
                if len(numbers) < 2:
                    return {"error": "除法需要至少两个数值"}
                result = numbers[0]
                for n in numbers[1:]:
                    if n == 0:
                        return {"error": "除数不能为零"}
                    result /= n
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "除法",
                    "inputs": (
                        [float(n) for n in numbers]
                        if not use_decimal
                        else [str(n) for n in numbers]
                    ),
                }

            elif operation == "power":
                if len(numbers) != 2:
                    return {"error": "幂运算需要恰好两个数值（底数和指数）"}
                base, exponent = numbers
                if use_decimal:
                    result = base**exponent
                else:
                    result = math.pow(float(base), float(exponent))
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "幂运算",
                    "base": float(base) if not use_decimal else str(base),
                    "exponent": float(exponent) if not use_decimal else str(exponent),
                }

            elif operation == "modulo":
                if len(numbers) != 2:
                    return {"error": "取模运算需要恰好两个数值"}
                dividend, divisor = numbers
                if divisor == 0:
                    return {"error": "除数不能为零"}
                result = dividend % divisor
                return {
                    "result": float(result) if not use_decimal else str(result),
                    "operation": "取模",
                    "dividend": float(dividend) if not use_decimal else str(dividend),
                    "divisor": float(divisor) if not use_decimal else str(divisor),
                }

            elif operation == "factorial":
                if len(numbers) != 1:
                    return {"error": "阶乘运算需要恰好一个数值"}
                n = numbers[0]
                if not isinstance(n, (int, Decimal)) and not n.is_integer():
                    return {"error": "阶乘运算需要非负整数"}
                n_int = int(n)
                if n_int < 0:
                    return {"error": "阶乘运算需要非负整数"}
                result = math.factorial(n_int)
                return {
                    "result": result,
                    "operation": "阶乘",
                    "input": n_int,
                }

            elif operation == "gcd":
                if len(numbers) < 2:
                    return {"error": "最大公约数需要至少两个数值"}
                integers = [
                    int(n)
                    for n in numbers
                    if isinstance(n, (int, float, Decimal)) and n == int(n)
                ]
                if len(integers) != len(numbers):
                    return {"error": "最大公约数需要整数"}
                result = integers[0]
                for n in integers[1:]:
                    result = math.gcd(result, n)
                return {
                    "result": result,
                    "operation": "最大公约数",
                    "inputs": integers,
                }

            elif operation == "lcm":
                if len(numbers) < 2:
                    return {"error": "最小公倍数需要至少两个数值"}
                integers = [
                    int(n)
                    for n in numbers
                    if isinstance(n, (int, float, Decimal)) and n == int(n)
                ]
                if len(integers) != len(numbers):
                    return {"error": "最小公倍数需要整数"}
                result = integers[0]
                for n in integers[1:]:
                    result = abs(result * n) // math.gcd(result, n)
                return {
                    "result": result,
                    "operation": "最小公倍数",
                    "inputs": integers,
                }

            else:
                return {"error": f"不支持的运算类型: {operation}"}

        except Exception as e:
            return {"error": f"算术运算出错: {str(e)}"}

    def mathematical_functions_tool(
        self,
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
            # 角度转换
            if function in ["sin", "cos", "tan"] and angle_unit == "degrees":
                value = math.radians(value)

            # 基本三角函数
            if function == "sin":
                result = math.sin(value)
                unit_note = (
                    f"（输入角度单位：{angle_unit}）" if angle_unit == "degrees" else ""
                )
            elif function == "cos":
                result = math.cos(value)
                unit_note = (
                    f"（输入角度单位：{angle_unit}）" if angle_unit == "degrees" else ""
                )
            elif function == "tan":
                result = math.tan(value)
                unit_note = (
                    f"（输入角度单位：{angle_unit}）" if angle_unit == "degrees" else ""
                )

            # 反三角函数
            elif function == "asin":
                if not -1 <= value <= 1:
                    return {"error": "asin的输入值必须在[-1, 1]范围内"}
                result = math.asin(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"（输出角度单位：{angle_unit}）"
            elif function == "acos":
                if not -1 <= value <= 1:
                    return {"error": "acos的输入值必须在[-1, 1]范围内"}
                result = math.acos(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"（输出角度单位：{angle_unit}）"
            elif function == "atan":
                result = math.atan(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
                unit_note = f"（输出角度单位：{angle_unit}）"

            # 双曲函数
            elif function == "sinh":
                result = math.sinh(value)
                unit_note = ""
            elif function == "cosh":
                result = math.cosh(value)
                unit_note = ""
            elif function == "tanh":
                result = math.tanh(value)
                unit_note = ""

            # 对数函数
            elif function == "log":
                if value <= 0:
                    return {"error": "对数函数的输入值必须大于0"}
                if base is None:
                    base = 10
                result = math.log(value, base)
                unit_note = f"（底数：{base}）"
            elif function == "log10":
                if value <= 0:
                    return {"error": "对数函数的输入值必须大于0"}
                result = math.log10(value)
                unit_note = "（常用对数，底数：10）"
            elif function == "ln":
                if value <= 0:
                    return {"error": "自然对数函数的输入值必须大于0"}
                result = math.log(value)
                unit_note = "（自然对数，底数：e）"

            # 根式和指数函数
            elif function == "sqrt":
                if value < 0:
                    return {"error": "平方根函数的输入值不能为负数"}
                result = math.sqrt(value)
                unit_note = ""
            elif function == "cbrt":
                result = math.pow(value, 1 / 3)
                unit_note = ""
            elif function == "exp":
                result = math.exp(value)
                unit_note = ""

            # 其他数学函数
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
                    f"（精度：{precision}位小数）" if precision is not None else ""
                )

            else:
                return {"error": f"不支持的数学函数: {function}"}

            # 格式化结果
            if precision is not None and function not in ["ceil", "floor", "round"]:
                result = round(result, precision)

            return {
                "result": result,
                "function": function,
                "input": value,
                "note": unit_note,
            }

        except Exception as e:
            return {"error": f"数学函数计算出错: {str(e)}"}

    def number_converter_tool(
        self,
        number: Union[str, float, int],
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
            if operation == "convert":
                # 进制转换
                if from_base < 2 or from_base > 36 or to_base < 2 or to_base > 36:
                    return {"error": "进制必须在2-36范围内"}

                # 转换为十进制
                if from_base != 10:
                    try:
                        decimal_value = int(str(number), from_base)
                    except ValueError:
                        return {
                            "error": f"无法将'{number}'从{from_base}进制转换为十进制"
                        }
                else:
                    decimal_value = int(float(number))

                # 从十进制转换到目标进制
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
                # 数值格式化
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
                    "operation": "格式化",
                }

            elif operation == "scientific":
                # 科学记数法
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
                # 转换为分数
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
                    return {"error": "无法转换为分数形式"}

            else:
                return {"error": f"不支持的操作类型: {operation}"}

        except Exception as e:
            return {"error": f"数值转换出错: {str(e)}"}

    def unit_converter_tool(
        self,
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
            # 定义转换因子（全部转换为基础单位）
            conversions = {
                "length": {  # 基础单位：米(m)
                    "mm": 0.001,
                    "cm": 0.01,
                    "m": 1,
                    "km": 1000,
                    "in": 0.0254,
                    "ft": 0.3048,
                    "yd": 0.9144,
                    "mi": 1609.34,
                },
                "weight": {  # 基础单位：千克(kg)
                    "mg": 0.000001,
                    "g": 0.001,
                    "kg": 1,
                    "t": 1000,
                    "oz": 0.0283495,
                    "lb": 0.453592,
                    "st": 6.35029,
                },
                "temperature": {  # 特殊处理
                    "celsius": "celsius",
                    "fahrenheit": "fahrenheit",
                    "kelvin": "kelvin",
                },
                "area": {  # 基础单位：平方米(m²)
                    "mm2": 0.000001,
                    "cm2": 0.0001,
                    "m2": 1,
                    "km2": 1000000,
                    "in2": 0.00064516,
                    "ft2": 0.092903,
                    "yd2": 0.836127,
                },
                "volume": {  # 基础单位：升(L)
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
                "time": {  # 基础单位：秒(s)
                    "ms": 0.001,
                    "s": 1,
                    "min": 60,
                    "h": 3600,
                    "day": 86400,
                    "week": 604800,
                    "month": 2629746,
                    "year": 31556952,
                },
                "speed": {  # 基础单位：米/秒(m/s)
                    "m/s": 1,
                    "km/h": 0.277778,
                    "mph": 0.44704,
                    "knot": 0.514444,
                },
                "energy": {  # 基础单位：焦耳(J)
                    "j": 1,
                    "kj": 1000,
                    "cal": 4.184,
                    "kcal": 4184,
                    "kwh": 3600000,
                },
            }

            if unit_type not in conversions:
                return {"error": f"不支持的单位类型: {unit_type}"}

            unit_map = conversions[unit_type]

            # 温度转换特殊处理
            if unit_type == "temperature":
                result = self._convert_temperature(value, from_unit, to_unit)
                if isinstance(result, dict) and "error" in result:
                    return result
            else:
                if from_unit not in unit_map or to_unit not in unit_map:
                    return {"error": f"不支持的单位: {from_unit} 或 {to_unit}"}

                # 转换为基础单位，然后转换为目标单位
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
            return {"error": f"单位转换出错: {str(e)}"}

    def _convert_temperature(
        self, value: float, from_unit: str, to_unit: str
    ) -> Union[float, Dict[str, str]]:
        """温度转换辅助函数"""
        try:
            # 统一单位名称
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()

            # 首先转换为摄氏度
            if from_unit == "celsius":
                celsius = value
            elif from_unit == "fahrenheit":
                celsius = (value - 32) * 5 / 9
            elif from_unit == "kelvin":
                celsius = value - 273.15
            else:
                return {"error": f"不支持的温度单位: {from_unit}"}

            # 从摄氏度转换为目标单位
            if to_unit == "celsius":
                result = celsius
            elif to_unit == "fahrenheit":
                result = celsius * 9 / 5 + 32
            elif to_unit == "kelvin":
                result = celsius + 273.15
            else:
                return {"error": f"不支持的温度单位: {to_unit}"}

            return result

        except Exception as e:
            return {"error": f"温度转换出错: {str(e)}"}

    def precision_calculator_tool(
        self,
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
            # 设置精度和舍入模式
            getcontext().prec = precision_digits + 10  # 额外精度避免舍入误差

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

            # 转换为高精度小数
            decimal_numbers = [Decimal(str(n)) for n in numbers]

            if operation == "add":
                result = sum(decimal_numbers)
            elif operation == "subtract":
                if len(decimal_numbers) < 2:
                    return {"error": "减法需要至少两个数值"}
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    result -= n
            elif operation == "multiply":
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    result *= n
            elif operation == "divide":
                if len(decimal_numbers) < 2:
                    return {"error": "除法需要至少两个数值"}
                result = decimal_numbers[0]
                for n in decimal_numbers[1:]:
                    if n == 0:
                        return {"error": "除数不能为零"}
                    result /= n
            elif operation == "power":
                if len(decimal_numbers) != 2:
                    return {"error": "幂运算需要恰好两个数值"}
                result = decimal_numbers[0] ** decimal_numbers[1]
            elif operation == "sqrt":
                if len(decimal_numbers) != 1:
                    return {"error": "平方根运算需要恰好一个数值"}
                if decimal_numbers[0] < 0:
                    return {"error": "平方根的输入值不能为负数"}
                result = decimal_numbers[0].sqrt()
            else:
                return {"error": f"不支持的运算类型: {operation}"}

            # 格式化结果
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
            return {"error": f"高精度计算出错: {str(e)}"}

    def number_properties_tool(
        self,
        number: Union[int, float],
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
            if isinstance(number, float) and not number.is_integer():
                return {"error": "某些分析功能仅支持整数"}

            n = (
                int(number)
                if isinstance(number, float) and number.is_integer()
                else number
            )
            result = {"number": n}

            if analysis_type in ["comprehensive", "properties"]:
                # 基本属性
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
                # 质数检验
                result["is_prime"] = self._is_prime(n)

                # 质因数分解
                if n <= 10000:  # 限制较大数的分解以避免超时
                    result["prime_factors"] = self._prime_factorization(n)

            if (
                analysis_type in ["comprehensive", "divisors"]
                and isinstance(n, int)
                and 0 < n <= 1000
            ):
                # 因数分析
                divisors = self._get_divisors(n)
                result.update(
                    {
                        "divisors": divisors,
                        "divisor_count": len(divisors),
                        "sum_of_divisors": sum(divisors),
                        "is_perfect_number": sum(divisors[:-1])
                        == n,  # 除自身外的因数和等于自身
                    }
                )

            if analysis_type == "comprehensive" and isinstance(n, int):
                # 数字根
                result["digital_root"] = self._digital_root(abs(n))

                # 各位数字和
                result["digit_sum"] = sum(int(digit) for digit in str(abs(n)))

            return result

        except Exception as e:
            return {"error": f"数值属性分析出错: {str(e)}"}

    def _is_prime(self, n: int) -> bool:
        """判断是否为质数"""
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
        """质因数分解"""
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
        """获取所有因数"""
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def _digital_root(self, n: int) -> int:
        """计算数字根"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n
