# -*- coding: utf-8 -*-
"""
金融数学计算模块
提供金融计算和分析工具
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional


class FinancialCalculator:
    """金融计算器类"""

    def __init__(self):
        pass

    def financial_calculator_tool(
        self,
        operation: str,
        principal: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[int] = None,
        cash_flows: Optional[List[float]] = None,
        initial_investment: Optional[float] = None,
        payment: Optional[float] = None,
        periods: Optional[int] = None,
        future_value: Optional[float] = None,
        present_value: Optional[float] = None,
        annual_rate: Optional[float] = None,
        payments_per_year: int = 12,
        risk_free_rate: Optional[float] = None,
        returns: Optional[List[float]] = None,
        prices: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        金融计算工具

        Args:
            operation: 操作类型
            principal: 本金
            rate: 利率
            time: 时间
            cash_flows: 现金流
            initial_investment: 初始投资
            payment: 支付金额
            periods: 期数
            future_value: 终值
            present_value: 现值
            annual_rate: 年利率
            payments_per_year: 每年支付次数
            risk_free_rate: 无风险利率
            returns: 收益率序列
            prices: 价格序列
        """
        try:
            # === 参数基本校验，无自动隐式映射，保持语义清晰 ===
            if operation == "compound_interest":
                return self._compound_interest(principal, rate, time)
            elif operation == "simple_interest":
                return self._simple_interest(principal, rate, time)
            elif operation == "present_value":
                return self._present_value_calculation(future_value, rate, time)
            elif operation == "future_value":
                # 若提供 present_value（单笔现值）
                if present_value is not None:
                    return self._future_value_calculation(present_value, rate, time)

                # 兼容性优化:
                # 若未提供 present_value 但提供 payment -> 计算年金未来价值 (期末支付)
                if payment is not None:
                    # 优先使用 rate；若未提供则尝试 annual_rate / payments_per_year
                    periodic_rate = rate
                    if periodic_rate is None and annual_rate is not None:
                        periodic_rate = annual_rate / payments_per_year

                    if periodic_rate is None:
                        return {"error": "计算年金未来价值需要提供 rate 或 annual_rate"}

                    fv = (
                        payment * ((1 + periodic_rate) ** periods - 1) / periodic_rate
                        if periodic_rate != 0
                        else payment * periods
                    )
                    return {
                        "operation": "future_value_annuity",
                        "payment": payment,
                        "rate": periodic_rate,
                        "periods": periods,
                        "future_value": round(fv, 2),
                        "formula": "PMT * ((1+r)^n - 1)/r",
                    }

                return {
                    "error": "future_value 需要 present_value，或使用 annuity 操作来计算分期缴款的终值"
                }
            elif operation == "annuity":
                return self._annuity_calculation(payment, rate, periods)
            elif operation == "npv":
                return self._net_present_value(cash_flows, rate)
            elif operation == "irr":
                return self._internal_rate_of_return(cash_flows)
            elif operation == "loan_payment":
                return self._loan_payment(
                    principal, rate, annual_rate, periods, payments_per_year
                )
            elif operation == "bond_pricing":
                return self._bond_pricing(principal, rate, annual_rate, periods)
            elif operation == "portfolio_metrics":
                return self._portfolio_metrics(returns, risk_free_rate)
            elif operation == "volatility":
                return self._volatility_calculation(prices or returns)
            elif operation == "sharpe_ratio":
                return self._sharpe_ratio(returns, risk_free_rate)
            else:
                return {"error": f"不支持的操作: {operation}"}

        except Exception as e:
            return {"error": f"金融计算错误: {str(e)}"}

    def _compound_interest(
        self, principal: float, rate: float, time: int
    ) -> Dict[str, Any]:
        """复利计算"""
        if principal <= 0 or rate < 0 or time < 0:
            return {"error": "参数值无效"}

        # 年复利
        amount = principal * (1 + rate) ** time
        interest = amount - principal

        # 月复利
        monthly_rate = rate / 12
        monthly_amount = principal * (1 + monthly_rate) ** (time * 12)
        monthly_interest = monthly_amount - principal

        # 连续复利
        continuous_amount = principal * math.exp(rate * time)
        continuous_interest = continuous_amount - principal

        return {
            "operation": "compound_interest",
            "principal": principal,
            "annual_rate": rate,
            "time_years": time,
            "annual_compounding": {
                "final_amount": round(amount, 2),
                "interest_earned": round(interest, 2),
                "effective_rate": round((amount / principal - 1), 4),
            },
            "monthly_compounding": {
                "final_amount": round(monthly_amount, 2),
                "interest_earned": round(monthly_interest, 2),
                "effective_rate": round((monthly_amount / principal - 1), 4),
            },
            "continuous_compounding": {
                "final_amount": round(continuous_amount, 2),
                "interest_earned": round(continuous_interest, 2),
                "effective_rate": round((continuous_amount / principal - 1), 4),
            },
        }

    def _simple_interest(
        self, principal: float, rate: float, time: int
    ) -> Dict[str, Any]:
        """单利计算"""
        if principal <= 0 or rate < 0 or time < 0:
            return {"error": "参数值无效"}

        interest = principal * rate * time
        amount = principal + interest

        return {
            "operation": "simple_interest",
            "principal": principal,
            "rate": rate,
            "time": time,
            "interest": round(interest, 2),
            "final_amount": round(amount, 2),
        }

    def _present_value_calculation(
        self, future_value: float, rate: float, time: int
    ) -> Dict[str, Any]:
        """现值计算"""
        if future_value <= 0 or rate < 0 or time < 0:
            return {"error": "参数值无效"}

        present_value = future_value / (1 + rate) ** time
        discount = future_value - present_value

        return {
            "operation": "present_value",
            "future_value": future_value,
            "discount_rate": rate,
            "time_periods": time,
            "present_value": round(present_value, 2),
            "discount_amount": round(discount, 2),
        }

    def _future_value_calculation(
        self, present_value: float, rate: float, time: int
    ) -> Dict[str, Any]:
        """终值计算"""
        if present_value <= 0 or rate < 0 or time < 0:
            return {"error": "参数值无效"}

        future_value = present_value * (1 + rate) ** time
        growth = future_value - present_value

        return {
            "operation": "future_value",
            "present_value": present_value,
            "growth_rate": rate,
            "time_periods": time,
            "future_value": round(future_value, 2),
            "growth_amount": round(growth, 2),
        }

    def _annuity_calculation(
        self, payment: float, rate: float, periods: int
    ) -> Dict[str, Any]:
        """年金计算"""
        if payment <= 0 or rate < 0 or periods <= 0:
            return {"error": "参数值无效"}

        # 普通年金现值
        if rate == 0:
            pv_ordinary = payment * periods
        else:
            pv_ordinary = payment * (1 - (1 + rate) ** (-periods)) / rate

        # 预付年金现值
        pv_due = pv_ordinary * (1 + rate)

        # 普通年金终值
        if rate == 0:
            fv_ordinary = payment * periods
        else:
            fv_ordinary = payment * ((1 + rate) ** periods - 1) / rate

        # 预付年金终值
        fv_due = fv_ordinary * (1 + rate)

        return {
            "operation": "annuity",
            "payment": payment,
            "rate": rate,
            "periods": periods,
            "ordinary_annuity": {
                "present_value": round(pv_ordinary, 2),
                "future_value": round(fv_ordinary, 2),
            },
            "annuity_due": {
                "present_value": round(pv_due, 2),
                "future_value": round(fv_due, 2),
            },
        }

    def _net_present_value(
        self, cash_flows: List[float], rate: float
    ) -> Dict[str, Any]:
        """净现值计算"""
        if not cash_flows or rate < 0:
            return {"error": "参数值无效"}

        npv = 0
        present_values = []

        for i, cf in enumerate(cash_flows):
            pv = cf / (1 + rate) ** i
            present_values.append(round(pv, 2))
            npv += pv

        return {
            "operation": "npv",
            "cash_flows": cash_flows,
            "discount_rate": rate,
            "present_values": present_values,
            "net_present_value": round(npv, 2),
            "decision": "接受项目" if npv > 0 else "拒绝项目",
        }

    def _internal_rate_of_return(self, cash_flows: List[float]) -> Dict[str, Any]:
        """内部收益率计算（牛顿法）"""
        if not cash_flows or len(cash_flows) < 2:
            return {"error": "需要至少两个现金流"}

        def npv_function(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

        def npv_derivative(rate):
            return sum(
                -i * cf / (1 + rate) ** (i + 1)
                for i, cf in enumerate(cash_flows)
                if i > 0
            )

        # 牛顿法求解
        rate = 0.1  # 初始猜测
        tolerance = 1e-6
        max_iterations = 100

        for iteration in range(max_iterations):
            npv = npv_function(rate)
            if abs(npv) < tolerance:
                break

            derivative = npv_derivative(rate)
            if abs(derivative) < tolerance:
                return {"error": "无法收敛到解"}

            rate = rate - npv / derivative

        if iteration == max_iterations - 1:
            return {"error": "达到最大迭代次数"}

        return {
            "operation": "irr",
            "cash_flows": cash_flows,
            "internal_rate_of_return": round(rate, 6),
            "irr_percentage": round(rate * 100, 2),
            "iterations": iteration + 1,
            "npv_at_irr": round(npv_function(rate), 6),
        }

    def _loan_payment(
        self,
        principal: float,
        periodic_rate: Optional[float],
        annual_rate: Optional[float],
        periods: Optional[int],
        payments_per_year: int = 12,
    ) -> Dict[str, Any]:
        """贷款还款计算"""
        # 参数验证
        if principal is None or principal <= 0:
            return {"error": "principal 必须为正数"}
        if periods is None or periods <= 0:
            return {"error": "periods 必须为正整数"}
        if periodic_rate is None and annual_rate is None:
            return {"error": "请提供 periodic_rate 或 annual_rate"}

        # 处理利率
        if periodic_rate is None:
            periodic_rate = annual_rate / payments_per_year

        if periodic_rate < 0:
            return {"error": "利率不能为负"}

        if periodic_rate == 0:
            payment = principal / periods
        else:
            payment = (
                principal
                * periodic_rate
                * (1 + periodic_rate) ** periods
                / ((1 + periodic_rate) ** periods - 1)
            )

        total_payment = payment * periods
        total_interest = total_payment - principal

        # 摊销表（前5期）
        amortization = []
        remaining_balance = principal

        for period in range(min(5, periods)):
            interest_payment = remaining_balance * periodic_rate
            principal_payment = payment - interest_payment
            remaining_balance -= principal_payment

            amortization.append(
                {
                    "period": period + 1,
                    "payment": round(payment, 2),
                    "interest": round(interest_payment, 2),
                    "principal": round(principal_payment, 2),
                    "balance": round(remaining_balance, 2),
                }
            )

        return {
            "operation": "loan_payment",
            "loan_amount": principal,
            "periodic_rate": periodic_rate,
            "annual_rate": annual_rate,
            "number_of_payments": periods,
            "payment_per_period": round(payment, 2),
            "total_payments": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "amortization_schedule": amortization,
        }

    def _bond_pricing(
        self, face_value: float, coupon_rate: float, yield_rate: float, periods: int
    ) -> Dict[str, Any]:
        """债券定价"""
        if face_value <= 0 or coupon_rate < 0 or yield_rate < 0 or periods <= 0:
            return {"error": "参数值无效"}

        coupon_payment = face_value * coupon_rate

        # 现值计算
        coupon_pv = 0
        if yield_rate == 0:
            coupon_pv = coupon_payment * periods
        else:
            coupon_pv = (
                coupon_payment * (1 - (1 + yield_rate) ** (-periods)) / yield_rate
            )

        face_value_pv = face_value / (1 + yield_rate) ** periods
        bond_price = coupon_pv + face_value_pv

        # 其他指标
        current_yield = coupon_payment / bond_price if bond_price > 0 else 0

        return {
            "operation": "bond_pricing",
            "face_value": face_value,
            "coupon_rate": coupon_rate,
            "yield_to_maturity": yield_rate,
            "periods_to_maturity": periods,
            "coupon_payment": round(coupon_payment, 2),
            "bond_price": round(bond_price, 2),
            "current_yield": round(current_yield, 4),
            "premium_discount": round(bond_price - face_value, 2),
        }

    def _portfolio_metrics(
        self, returns: List[float], risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """投资组合指标"""
        if not returns:
            return {"error": "收益率数据不能为空"}

        returns_array = np.array(returns)

        # 基本统计
        mean_return = np.mean(returns_array)
        std_dev = np.std(returns_array, ddof=1)
        variance = np.var(returns_array, ddof=1)

        # 夏普比率
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0

        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # VaR (5%)
        var_5 = np.percentile(returns_array, 5)

        return {
            "operation": "portfolio_metrics",
            "number_of_periods": len(returns),
            "mean_return": round(mean_return, 6),
            "volatility": round(std_dev, 6),
            "variance": round(variance, 6),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_drawdown, 6),
            "var_5_percent": round(var_5, 6),
            "annualized_return": round(mean_return * 252, 4),  # 假设252个交易日
            "annualized_volatility": round(std_dev * math.sqrt(252), 4),
        }

    def _volatility_calculation(self, data: List[float]) -> Dict[str, Any]:
        """波动率计算"""
        if not data or len(data) < 2:
            return {"error": "数据不足"}

        data_array = np.array(data)

        # 如果是价格数据，计算收益率
        if all(x > 0 for x in data):
            returns = np.diff(np.log(data_array))
        else:
            returns = data_array

        # 历史波动率
        volatility = np.std(returns, ddof=1)

        # 年化波动率
        annualized_volatility = volatility * math.sqrt(252)

        return {
            "operation": "volatility",
            "data_points": len(data),
            "returns": returns.tolist(),
            "daily_volatility": round(volatility, 6),
            "annualized_volatility": round(annualized_volatility, 4),
            "volatility_percentage": round(annualized_volatility * 100, 2),
        }

    def _sharpe_ratio(
        self, returns: List[float], risk_free_rate: float
    ) -> Dict[str, Any]:
        """夏普比率计算"""
        if not returns:
            return {"error": "收益率数据不能为空"}

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)

        sharpe_ratio = (
            mean_excess_return / std_excess_return if std_excess_return > 0 else 0
        )

        return {
            "operation": "sharpe_ratio",
            "returns": returns,
            "risk_free_rate": risk_free_rate,
            "excess_returns": excess_returns.tolist(),
            "mean_excess_return": round(mean_excess_return, 6),
            "volatility_excess_return": round(std_excess_return, 6),
            "sharpe_ratio": round(sharpe_ratio, 4),
        }
