# -*- coding: utf-8 -*-
"""
Financial Mathematics Calculation Module
Provides financial calculation and analysis tools
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional


class FinancialCalculator:
    """Financial Calculator Class"""

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
        Financial Calculation Tool

        Args:
            operation: Operation type
            principal: Principal amount
            rate: Interest rate
            time: Time period
            cash_flows: Cash flows
            initial_investment: Initial investment
            payment: Payment amount
            periods: Number of periods
            future_value: Future value
            present_value: Present value
            annual_rate: Annual interest rate
            payments_per_year: Number of payments per year
            risk_free_rate: Risk-free rate
            returns: Returns sequence
            prices: Price sequence
        """
        try:
            # === Basic parameter validation, no automatic implicit mapping, maintaining semantic clarity ===
            if operation == "compound_interest":
                return self._compound_interest(principal, rate, time)
            elif operation == "simple_interest":
                return self._simple_interest(principal, rate, time)
            elif operation == "present_value":
                return self._present_value_calculation(future_value, rate, time)
            elif operation == "future_value":
                # If present_value is provided (single present value)
                if present_value is not None:
                    return self._future_value_calculation(present_value, rate, time)

                # Compatibility optimization:
                # If present_value is not provided but payment is provided -> calculate annuity future value (end-of-period payment)
                if payment is not None:
                    # Prioritize rate; if not provided, try annual_rate / payments_per_year
                    periodic_rate = rate
                    if periodic_rate is None and annual_rate is not None:
                        periodic_rate = annual_rate / payments_per_year

                    if periodic_rate is None:
                        return {
                            "error": "Calculating annuity future value requires providing rate or annual_rate"
                        }

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
                    "error": "future_value requires present_value, or use annuity operation to calculate installment payment future value"
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
                return {"error": f"Unsupported operation: {operation}"}

        except Exception as e:
            return {"error": f"Financial calculation error: {str(e)}"}

    def _compound_interest(
        self, principal: float, rate: float, time: int
    ) -> Dict[str, Any]:
        """Compound interest calculation"""
        if principal <= 0 or rate < 0 or time < 0:
            return {"error": "Invalid parameter values"}

        # Annual compounding
        amount = principal * (1 + rate) ** time
        interest = amount - principal

        # Monthly compounding
        monthly_rate = rate / 12
        monthly_amount = principal * (1 + monthly_rate) ** (time * 12)
        monthly_interest = monthly_amount - principal

        # Continuous compounding
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
        """Simple interest calculation"""
        if principal <= 0 or rate < 0 or time < 0:
            return {"error": "Invalid parameter values"}

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
        """Present value calculation"""
        if future_value <= 0 or rate < 0 or time < 0:
            return {"error": "Invalid parameter values"}

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
        """Future value calculation"""
        if present_value <= 0 or rate < 0 or time < 0:
            return {"error": "Invalid parameter values"}

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
        """Annuity calculation"""
        if payment <= 0 or rate < 0 or periods <= 0:
            return {"error": "Invalid parameter values"}

        # Ordinary annuity present value
        if rate == 0:
            pv_ordinary = payment * periods
        else:
            pv_ordinary = payment * (1 - (1 + rate) ** (-periods)) / rate

        # Annuity due present value
        pv_due = pv_ordinary * (1 + rate)

        # Ordinary annuity future value
        if rate == 0:
            fv_ordinary = payment * periods
        else:
            fv_ordinary = payment * ((1 + rate) ** periods - 1) / rate

        # Annuity due future value
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
        """Net present value calculation"""
        if not cash_flows or rate < 0:
            return {"error": "Invalid parameter values"}

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
            "decision": "Accept project" if npv > 0 else "Reject project",
        }

    def _internal_rate_of_return(self, cash_flows: List[float]) -> Dict[str, Any]:
        """Internal rate of return calculation (Newton's method)"""
        if not cash_flows or len(cash_flows) < 2:
            return {"error": "At least two cash flows are required"}

        def npv_function(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

        def npv_derivative(rate):
            return sum(
                -i * cf / (1 + rate) ** (i + 1)
                for i, cf in enumerate(cash_flows)
                if i > 0
            )

        # Newton's method solution
        rate = 0.1  # Initial guess
        tolerance = 1e-6
        max_iterations = 100

        for iteration in range(max_iterations):
            npv = npv_function(rate)
            if abs(npv) < tolerance:
                break

            derivative = npv_derivative(rate)
            if abs(derivative) < tolerance:
                return {"error": "Unable to converge to solution"}

            rate = rate - npv / derivative

        if iteration == max_iterations - 1:
            return {"error": "Maximum iterations reached"}

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
        """Loan payment calculation"""
        # Parameter validation
        if principal is None or principal <= 0:
            return {"error": "principal must be a positive number"}
        if periods is None or periods <= 0:
            return {"error": "periods must be a positive integer"}
        if periodic_rate is None and annual_rate is None:
            return {"error": "Please provide periodic_rate or annual_rate"}

        # Handle interest rate
        if periodic_rate is None:
            periodic_rate = annual_rate / payments_per_year

        if periodic_rate < 0:
            return {"error": "Interest rate cannot be negative"}

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

        # Amortization schedule (first 5 periods)
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
        """Bond pricing"""
        if face_value <= 0 or coupon_rate < 0 or yield_rate < 0 or periods <= 0:
            return {"error": "Invalid parameter values"}

        coupon_payment = face_value * coupon_rate

        # Present value calculation
        coupon_pv = 0
        if yield_rate == 0:
            coupon_pv = coupon_payment * periods
        else:
            coupon_pv = (
                coupon_payment * (1 - (1 + yield_rate) ** (-periods)) / yield_rate
            )

        face_value_pv = face_value / (1 + yield_rate) ** periods
        bond_price = coupon_pv + face_value_pv

        # Other indicators
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
        """Portfolio metrics"""
        if not returns:
            return {"error": "Returns data cannot be empty"}

        returns_array = np.array(returns)

        # Basic statistics
        mean_return = np.mean(returns_array)
        std_dev = np.std(returns_array, ddof=1)
        variance = np.var(returns_array, ddof=1)

        # Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0

        # Maximum drawdown
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
            "annualized_return": round(
                mean_return * 252, 4
            ),  # Assuming 252 trading days
            "annualized_volatility": round(std_dev * math.sqrt(252), 4),
        }

    def _volatility_calculation(self, data: List[float]) -> Dict[str, Any]:
        """Volatility calculation"""
        if not data or len(data) < 2:
            return {"error": "Insufficient data"}

        data_array = np.array(data)

        # If price data, calculate returns
        if all(x > 0 for x in data):
            returns = np.diff(np.log(data_array))
        else:
            returns = data_array

        # Historical volatility
        volatility = np.std(returns, ddof=1)

        # Annualized volatility
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
        """Sharpe ratio calculation"""
        if not returns:
            return {"error": "Returns data cannot be empty"}

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
