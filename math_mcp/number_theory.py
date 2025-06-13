# -*- coding: utf-8 -*-
"""
数论计算模块
提供高级数论计算功能
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple


class NumberTheoryCalculator:
    """数论计算器类"""

    def __init__(self):
        pass

    def number_theory_tool(
        self,
        operation: str,
        number: Optional[int] = None,
        numbers: Optional[List[int]] = None,
        modulus: Optional[int] = None,
        base: Optional[int] = None,
        exponent: Optional[int] = None,
        limit: Optional[int] = None,
        precision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        数论计算工具

        Args:
            operation: 操作类型
            number: 单个数字
            numbers: 数字列表
            modulus: 模数
            base: 底数
            exponent: 指数
            limit: 限制值
            precision: 计算精度
        """
        # 参数验证
        validation_result = self._validate_parameters(
            operation, number, numbers, modulus, base, exponent, limit
        )
        if validation_result is not None:
            return validation_result

        try:
            if operation == "prime_factorization":
                return self._prime_factorization(number)
            elif operation == "prime_test":
                return self._prime_test(number)
            elif operation == "generate_primes":
                return self._generate_primes(limit)
            elif operation == "modular_arithmetic":
                op_type = "add"  # 默认加法，可以通过扩展参数指定
                return self._modular_arithmetic(numbers, modulus, op_type)
            elif operation == "modular_exponentiation":
                return self._modular_exponentiation(base, exponent, modulus)
            elif operation == "extended_gcd":
                return self._extended_gcd(
                    numbers[0] if numbers else 0,
                    numbers[1] if numbers and len(numbers) > 1 else 0,
                )
            elif operation == "chinese_remainder":
                return self._chinese_remainder_theorem(
                    numbers[: len(numbers) // 2] if numbers else [],
                    numbers[len(numbers) // 2 :] if numbers else [],
                )
            elif operation == "euler_totient":
                return self._euler_totient(number)
            elif operation == "divisors":
                return self._find_divisors(number)
            elif operation == "perfect_number":
                return self._perfect_number_check(number)
            elif operation == "fibonacci":
                return self._fibonacci_sequence(limit)
            elif operation == "collatz":
                return self._collatz_sequence(number)
            elif operation == "carmichael_function":
                return self._carmichael_function(number)
            elif operation == "jacobi_symbol":
                return self._jacobi_symbol(
                    numbers[0] if numbers else 0,
                    numbers[1] if numbers and len(numbers) > 1 else 0,
                )
            elif operation == "quadratic_residue":
                return self._quadratic_residue(number, modulus)
            elif operation == "primitive_root":
                return self._primitive_root(number)
            elif operation == "continued_fraction":
                return self._continued_fraction(number, precision or 10)
            else:
                return {"error": f"不支持的操作: {operation}"}

        except Exception as e:
            return {"error": f"数论计算错误: {str(e)}"}

    def _validate_parameters(
        self,
        operation: str,
        number: Optional[int],
        numbers: Optional[List[int]],
        modulus: Optional[int],
        base: Optional[int],
        exponent: Optional[int],
        limit: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """参数验证"""
        valid_operations = [
            "prime_factorization",
            "prime_test",
            "generate_primes",
            "modular_arithmetic",
            "modular_exponentiation",
            "extended_gcd",
            "chinese_remainder",
            "euler_totient",
            "divisors",
            "perfect_number",
            "fibonacci",
            "collatz",
            "carmichael_function",
            "jacobi_symbol",
            "quadratic_residue",
            "primitive_root",
            "continued_fraction",
        ]

        if operation not in valid_operations:
            return {
                "error": f"无效的操作类型: {operation}，支持的操作: {valid_operations}"
            }

        # 检查必需参数
        if operation in [
            "prime_factorization",
            "prime_test",
            "euler_totient",
            "divisors",
            "perfect_number",
            "collatz",
            "carmichael_function",
            "primitive_root",
            "continued_fraction",
        ]:
            if number is None:
                return {"error": f"操作 {operation} 需要提供 number 参数"}
            if not isinstance(number, int) or number <= 0:
                return {"error": "number 必须是正整数"}

        if operation in ["generate_primes", "fibonacci"]:
            if limit is None:
                return {"error": f"操作 {operation} 需要提供 limit 参数"}
            if not isinstance(limit, int) or limit <= 0:
                return {"error": "limit 必须是正整数"}

        if operation == "modular_exponentiation":
            if base is None or exponent is None or modulus is None:
                return {"error": "模幂运算需要提供 base, exponent, modulus 参数"}
            if not all(isinstance(x, int) for x in [base, exponent, modulus]):
                return {"error": "base, exponent, modulus 必须是整数"}
            if modulus <= 0:
                return {"error": "modulus 必须是正整数"}
            if exponent < 0:
                return {"error": "exponent 必须是非负整数"}

        if operation in ["extended_gcd", "jacobi_symbol"]:
            if numbers is None or len(numbers) < 2:
                return {"error": f"操作 {operation} 需要提供至少2个数字"}
            if not all(isinstance(x, int) for x in numbers):
                return {"error": "numbers 中的所有元素必须是整数"}

        if operation == "quadratic_residue":
            if number is None or modulus is None:
                return {"error": "二次剩余检查需要提供 number 和 modulus 参数"}
            if not isinstance(modulus, int) or modulus <= 0:
                return {"error": "modulus 必须是正整数"}

        return None

    def _prime_factorization(self, n: int) -> Dict[str, Any]:
        """素因数分解"""
        if n <= 1:
            return {"error": "数字必须大于1"}

        factors = []
        original_n = n

        # 处理2的因子
        while n % 2 == 0:
            factors.append(2)
            n //= 2

        # 处理奇数因子
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2

        # 如果n是大于2的素数
        if n > 2:
            factors.append(n)

        # 统计因子频次
        factor_counts = {}
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

        return {
            "operation": "prime_factorization",
            "number": original_n,
            "prime_factors": factors,
            "factor_counts": factor_counts,
            "unique_factors": list(factor_counts.keys()),
            "factorization_string": " × ".join(
                [f"{f}^{c}" if c > 1 else str(f) for f, c in factor_counts.items()]
            ),
            "is_prime": len(factors) == 1,
            "number_of_prime_factors": len(factors),
            "number_of_distinct_prime_factors": len(factor_counts),
        }

    def _prime_test(self, n: int) -> Dict[str, Any]:
        """素数检测（Miller-Rabin算法）"""
        if n < 2:
            return {
                "number": n,
                "is_prime": False,
                "reason": "小于2",
                "certainty": "确定",
            }
        if n == 2:
            return {
                "number": n,
                "is_prime": True,
                "reason": "2是素数",
                "certainty": "确定",
            }
        if n % 2 == 0:
            return {
                "number": n,
                "is_prime": False,
                "reason": "偶数",
                "certainty": "确定",
            }

        # 小素数试除
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if n == p:
                return {
                    "number": n,
                    "is_prime": True,
                    "reason": f"{p}是已知素数",
                    "certainty": "确定",
                }
            if n % p == 0:
                return {
                    "number": n,
                    "is_prime": False,
                    "reason": f"被{p}整除",
                    "certainty": "确定",
                }

        # Miller-Rabin素性测试
        def __miller_rabin(n, k=10):
            # 写n-1为d*2^r的形式
            r = 0
            d = n - 1
            while d % 2 == 0:
                r += 1
                d //= 2

            # 进行k轮测试
            for _ in range(k):
                a = random.randrange(2, n - 1)
                x = pow(a, d, n)

                if x == 1 or x == n - 1:
                    continue

                for _ in range(r - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    return False
            return True

        is_prime = __miller_rabin(n)

        return {
            "operation": "prime_test",
            "number": n,
            "is_prime": is_prime,
            "method": "Miller-Rabin",
            "certainty": "高概率正确" if is_prime else "确定合数",
            "test_rounds": 10,
        }

    def _generate_primes(self, limit: int) -> Dict[str, Any]:
        """生成素数（埃拉托斯特尼筛法）"""
        if limit < 2:
            return {"limit": limit, "primes": [], "count": 0}

        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False

        primes = [i for i in range(2, limit + 1) if sieve[i]]

        # 计算素数密度
        density = len(primes) / limit if limit > 0 else 0

        return {
            "operation": "generate_primes",
            "limit": limit,
            "primes": primes,
            "count": len(primes),
            "largest_prime": primes[-1] if primes else None,
            "density": round(density, 6),
            "prime_gaps": (
                [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]
                if len(primes) > 1
                else []
            ),
        }

    def _modular_arithmetic(
        self, numbers: List[int], modulus: int, operation: str = "add"
    ) -> Dict[str, Any]:
        """模运算"""
        if not numbers or modulus <= 0:
            return {"error": "无效参数"}

        result = numbers[0] % modulus

        if operation == "add":
            for num in numbers[1:]:
                result = (result + num) % modulus
        elif operation == "multiply":
            for num in numbers[1:]:
                result = (result * num) % modulus
        elif operation == "subtract" and len(numbers) >= 2:
            result = (numbers[0] - numbers[1]) % modulus
        elif operation == "power" and len(numbers) >= 2:
            result = pow(numbers[0], numbers[1], modulus)

        return {
            "operation": "modular_arithmetic",
            "numbers": numbers,
            "modulus": modulus,
            "arithmetic_operation": operation,
            "result": result,
        }

    def _modular_exponentiation(
        self, base: int, exponent: int, modulus: int
    ) -> Dict[str, Any]:
        """模幂运算"""
        if modulus <= 0:
            return {"error": "模数必须为正数"}

        result = pow(base, exponent, modulus)

        return {
            "operation": "modular_exponentiation",
            "base": base,
            "exponent": exponent,
            "modulus": modulus,
            "result": result,
            "expression": f"{base}^{exponent} ≡ {result} (mod {modulus})",
        }

    def _extended_gcd(self, a: int, b: int) -> Dict[str, Any]:
        """扩展欧几里得算法"""
        original_a, original_b = a, b
        old_r, r = a, b
        old_s, s = 1, 0
        old_t, t = 0, 1

        steps = []

        while r != 0:
            quotient = old_r // r
            steps.append({"quotient": quotient, "remainder": r, "s": s, "t": t})
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t

        gcd = old_r

        return {
            "operation": "extended_gcd",
            "a": original_a,
            "b": original_b,
            "gcd": gcd,
            "x": old_s,
            "y": old_t,
            "equation": f"{original_a} × {old_s} + {original_b} × {old_t} = {gcd}",
            "steps": steps,
            "coprime": gcd == 1,
        }

    def _chinese_remainder_theorem(
        self, remainders: List[int], moduli: List[int]
    ) -> Dict[str, Any]:
        """中国剩余定理"""
        if len(remainders) != len(moduli):
            return {"error": "余数和模数的数量必须相等"}

        if len(remainders) == 0:
            return {"error": "需要至少一个方程"}

        # 检查模数是否两两互质
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if math.gcd(moduli[i], moduli[j]) != 1:
                    return {"error": f"模数{moduli[i]}和{moduli[j]}不互质"}

        # 中国剩余定理求解
        total = 0
        prod = 1
        for m in moduli:
            prod *= m

        for i in range(len(remainders)):
            p = prod // moduli[i]
            total += remainders[i] * self._mod_inverse(p, moduli[i]) * p

        result = total % prod

        return {
            "operation": "chinese_remainder",
            "system": [
                f"x ≡ {remainders[i]} (mod {moduli[i]})" for i in range(len(remainders))
            ],
            "solution": result,
            "modulus": prod,
            "verification": [
                f"{result} ≡ {result % moduli[i]} (mod {moduli[i]})"
                for i in range(len(moduli))
            ],
        }

    def _mod_inverse(self, a: int, m: int) -> int:
        """模逆元"""
        gcd, x, _ = self._extended_gcd_internal(a, m)
        if gcd != 1:
            raise ValueError("模逆元不存在")
        return (x % m + m) % m

    def _extended_gcd_internal(self, a: int, b: int) -> Tuple[int, int, int]:
        """内部扩展欧几里得算法"""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = self._extended_gcd_internal(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def _euler_totient(self, n: int) -> Dict[str, Any]:
        """欧拉函数"""
        if n <= 0:
            return {"error": "数字必须为正数"}

        result = n
        original_n = n

        # 获取素因数
        factors = set()
        temp = n

        for i in range(2, int(math.sqrt(temp)) + 1):
            if temp % i == 0:
                factors.add(i)
                while temp % i == 0:
                    temp //= i

        if temp > 1:
            factors.add(temp)

        # 计算欧拉函数值
        for p in factors:
            result = result * (p - 1) // p

        return {
            "operation": "euler_totient",
            "number": original_n,
            "euler_totient": result,
            "prime_factors": list(factors),
            "formula": f"φ({original_n}) = {result}",
            "relatively_prime_count": result,
        }

    def _carmichael_function(self, n: int) -> Dict[str, Any]:
        """卡迈克尔函数"""
        if n <= 0:
            return {"error": "数字必须为正数"}

        if n == 1:
            return {"number": 1, "carmichael": 1}

        # 质因数分解
        factors = self._prime_factorization(n)
        if "error" in factors:
            return factors

        factor_counts = factors["factor_counts"]
        lambda_values = []

        for p, k in factor_counts.items():
            if p == 2:
                if k == 1:
                    lambda_values.append(1)
                elif k == 2:
                    lambda_values.append(2)
                else:
                    lambda_values.append(2 ** (k - 2))
            else:
                lambda_values.append((p - 1) * (p ** (k - 1)))

        # 计算最小公倍数
        result = lambda_values[0]
        for val in lambda_values[1:]:
            result = result * val // math.gcd(result, val)

        return {
            "operation": "carmichael_function",
            "number": n,
            "carmichael": result,
            "prime_powers": {p: k for p, k in factor_counts.items()},
            "lambda_values": lambda_values,
        }

    def _jacobi_symbol(self, a: int, n: int) -> Dict[str, Any]:
        """雅可比符号"""
        if n <= 0 or n % 2 == 0:
            return {"error": "n必须是正奇数"}

        original_a, original_n = a, n
        result = 1

        a = a % n

        while a != 0:
            while a % 2 == 0:
                a //= 2
                if n % 8 in [3, 5]:
                    result = -result

            a, n = n, a

            if a % 4 == 3 and n % 4 == 3:
                result = -result

            a = a % n

        if n == 1:
            jacobi_result = result
        else:
            jacobi_result = 0

        return {
            "operation": "jacobi_symbol",
            "a": original_a,
            "n": original_n,
            "jacobi_symbol": jacobi_result,
            "interpretation": {
                1: "a是模n的二次剩余（或n=1）",
                -1: "a是模n的二次非剩余",
                0: "gcd(a,n) > 1",
            }[jacobi_result],
        }

    def _quadratic_residue(self, a: int, p: int) -> Dict[str, Any]:
        """二次剩余检查"""
        if not self._is_prime_simple(p):
            return {"error": "p必须是素数"}

        a = a % p

        if a == 0:
            legendre = 0
        else:
            legendre = pow(a, (p - 1) // 2, p)
            if legendre == p - 1:
                legendre = -1

        is_residue = legendre == 1

        # 如果是二次剩余，尝试找到平方根
        roots = []
        if is_residue:
            for x in range(p):
                if (x * x) % p == a:
                    roots.append(x)

        return {
            "operation": "quadratic_residue",
            "a": a,
            "p": p,
            "is_quadratic_residue": is_residue,
            "legendre_symbol": legendre if legendre != 0 else 0,
            "square_roots": roots,
        }

    def _primitive_root(self, n: int) -> Dict[str, Any]:
        """寻找原根"""
        if n <= 0:
            return {"error": "数字必须为正数"}

        # 检查n是否有原根
        if not self._has_primitive_root(n):
            return {"error": f"{n}没有原根"}

        phi_n = self._euler_totient(n)["euler_totient"]

        # 找到φ(n)的所有素因数
        factors = set()
        temp = phi_n
        for i in range(2, int(math.sqrt(temp)) + 1):
            if temp % i == 0:
                factors.add(i)
                while temp % i == 0:
                    temp //= i
        if temp > 1:
            factors.add(temp)

        # 寻找原根
        primitive_roots = []
        for g in range(1, min(n, 100)):  # 限制搜索范围
            if math.gcd(g, n) == 1:
                is_primitive = True
                for p in factors:
                    if pow(g, phi_n // p, n) == 1:
                        is_primitive = False
                        break
                if is_primitive:
                    primitive_roots.append(g)

        return {
            "operation": "primitive_root",
            "number": n,
            "euler_phi": phi_n,
            "primitive_roots": primitive_roots[:10],  # 只返回前10个
            "count_primitive_roots": len(primitive_roots),
            "has_primitive_root": len(primitive_roots) > 0,
        }

    def _continued_fraction(self, number: float, max_terms: int = 10) -> Dict[str, Any]:
        """连分数表示"""
        if max_terms <= 0:
            return {"error": "项数必须为正数"}

        original_number = number
        terms = []

        for _ in range(max_terms):
            integer_part = int(number)
            terms.append(integer_part)

            fractional_part = number - integer_part
            if abs(fractional_part) < 1e-10:  # 接近0
                break

            number = 1 / fractional_part

        # 计算连分数的收敛值
        convergents = []
        for i in range(len(terms)):
            if i == 0:
                convergents.append((terms[0], 1))
            elif i == 1:
                convergents.append((terms[1] * terms[0] + 1, terms[1]))
            else:
                p_prev2, q_prev2 = convergents[i - 2]
                p_prev1, q_prev1 = convergents[i - 1]
                p_curr = terms[i] * p_prev1 + p_prev2
                q_curr = terms[i] * q_prev1 + q_prev2
                convergents.append((p_curr, q_curr))

        return {
            "operation": "continued_fraction",
            "number": original_number,
            "terms": terms,
            "convergents": [
                {"numerator": p, "denominator": q, "value": p / q}
                for p, q in convergents
            ],
            "representation": " + ".join(
                [f"1/({term})" if i > 0 else str(term) for i, term in enumerate(terms)]
            ),
        }

    def _has_primitive_root(self, n: int) -> bool:
        """检查n是否有原根"""
        if n == 1 or n == 2 or n == 4:
            return True

        # n = p^k 或 n = 2*p^k，其中p是奇素数
        if n % 2 == 0:
            n //= 2

        # 检查是否为素数的幂
        for p in range(3, int(n**0.5) + 1, 2):
            if n % p == 0:
                temp = n
                while temp % p == 0:
                    temp //= p
                return temp == 1

        return self._is_prime_simple(n)

    def _is_prime_simple(self, n: int) -> bool:
        """简单素数检测"""
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

    def _find_divisors(self, n: int) -> Dict[str, Any]:
        """找出所有因数"""
        if n <= 0:
            return {"error": "数字必须为正数"}

        divisors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)

        divisors.sort()

        # 计算因数的一些统计信息
        sum_divisors = sum(divisors)
        proper_divisors = [d for d in divisors if d != n]
        sum_proper = sum(proper_divisors)

        return {
            "operation": "divisors",
            "number": n,
            "divisors": divisors,
            "count": len(divisors),
            "sum_of_divisors": sum_divisors,
            "proper_divisors": proper_divisors,
            "sum_of_proper_divisors": sum_proper,
            "is_perfect": sum_proper == n,
            "is_abundant": sum_proper > n,
            "is_deficient": sum_proper < n,
            "abundance": sum_proper - n,
        }

    def _perfect_number_check(self, n: int) -> Dict[str, Any]:
        """完数检查"""
        divisors_result = self._find_divisors(n)
        if "error" in divisors_result:
            return divisors_result

        sum_proper = divisors_result["sum_of_proper_divisors"]
        is_perfect = sum_proper == n

        return {
            "operation": "perfect_number",
            "number": n,
            "is_perfect": is_perfect,
            "proper_divisors": divisors_result["proper_divisors"],
            "sum_of_proper_divisors": sum_proper,
            "classification": (
                "完数" if is_perfect else ("盈数" if sum_proper > n else "亏数")
            ),
            "abundance": sum_proper - n,
        }

    def _fibonacci_sequence(self, n: int) -> Dict[str, Any]:
        """斐波那契数列"""
        if n <= 0:
            return {"limit": n, "sequence": [], "count": 0}

        sequence = []
        a, b = 0, 1

        while a <= n:
            sequence.append(a)
            a, b = b, a + b

        # 计算黄金比例近似值
        golden_ratio = None
        if len(sequence) > 1:
            golden_ratio = sequence[-1] / sequence[-2] if sequence[-2] != 0 else None

        return {
            "operation": "fibonacci",
            "limit": n,
            "sequence": sequence,
            "count": len(sequence),
            "largest_fibonacci": sequence[-1] if sequence else 0,
            "golden_ratio_approximation": golden_ratio,
        }

    def _collatz_sequence(self, n: int) -> Dict[str, Any]:
        """考拉兹猜想序列"""
        if n <= 0:
            return {"error": "数字必须为正数"}

        sequence = [n]
        steps = 0
        max_value = n

        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            sequence.append(n)
            steps += 1
            max_value = max(max_value, n)

            # 防止无限循环
            if steps > 10000:
                return {"error": "序列过长，可能存在问题"}

        return {
            "operation": "collatz",
            "starting_number": sequence[0],
            "sequence": (
                sequence
                if len(sequence) <= 100
                else sequence[:50] + ["..."] + sequence[-50:]
            ),
            "full_sequence_length": len(sequence),
            "steps_to_one": steps,
            "max_value": max_value,
            "trajectory_record": max_value > sequence[0],
        }
