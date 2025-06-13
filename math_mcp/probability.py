# -*- coding: utf-8 -*-
"""
Probability calculation module
Provides probability distribution and stochastic process calculation functions
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class ProbabilityCalculator:
    """Probability calculator class"""

    def __init__(self):
        self.e = math.e
        self.pi = math.pi

    def probability_calculator_tool(
        self,
        operation: str,
        distribution: Optional[str] = None,
        parameters: Optional[Dict[str, float]] = None,
        x_value: Optional[float] = None,
        x_values: Optional[List[float]] = None,
        probability: Optional[float] = None,
        n_samples: Optional[int] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Probability calculation tool

        Args:
            operation: Operation type
            distribution: Distribution type
            parameters: Distribution parameters
            x_value: Single x value
            x_values: List of x values
            probability: Probability value
            n_samples: Number of samples
            events: List of events
            data: Data list
        """
        try:
            if operation == "probability_mass":
                return self._probability_mass_function(
                    distribution, parameters, x_value
                )
            elif operation == "cumulative_distribution":
                return self._cumulative_distribution_function(
                    distribution, parameters, x_value
                )
            elif operation == "inverse_cdf":
                return self._inverse_cdf(distribution, parameters, probability)
            elif operation == "random_sampling":
                return self._random_sampling(distribution, parameters, n_samples)
            elif operation == "distribution_stats":
                return self._distribution_statistics(distribution, parameters)
            elif operation == "bayes_theorem":
                return self._bayes_theorem(events)
            elif operation == "combinatorics":
                return self._combinatorics_calculation(parameters)
            elif operation == "hypothesis_test":
                return self._hypothesis_testing(data, parameters)
            elif operation == "confidence_interval":
                return self._confidence_interval(data, parameters)
            elif operation == "monte_carlo":
                return self._monte_carlo_simulation(parameters, n_samples)
            else:
                return {"error": f"Unsupported operation: {operation}"}

        except Exception as e:
            return {"error": f"Probability calculation error: {str(e)}"}

    def _probability_mass_function(
        self, distribution: str, parameters: Dict[str, float], x: float
    ) -> Dict[str, Any]:
        """Probability mass function/probability density function"""
        if distribution == "normal":
            # Compatible with multiple parameter naming conventions
            mu = parameters.get("mean", parameters.get("mu", 0))
            sigma = parameters.get("std", parameters.get("sigma", 1))
            if sigma <= 0:
                return {"error": "Standard deviation must be positive"}
            pdf = (1 / (sigma * math.sqrt(2 * self.pi))) * math.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )

        elif distribution == "binomial":
            n = int(parameters.get("n", 10))
            p = parameters.get("p", 0.5)
            if not (0 <= p <= 1) or n < 0 or x < 0 or x > n:
                return {"error": "Invalid parameter values"}
            pdf = self._combination(n, int(x)) * (p**x) * ((1 - p) ** (n - x))

        elif distribution == "poisson":
            lambda_param = parameters.get("lambda", 1)
            if lambda_param <= 0 or x < 0:
                return {"error": "Invalid parameter values"}
            pdf = (lambda_param**x) * math.exp(-lambda_param) / math.factorial(int(x))

        elif distribution == "exponential":
            lambda_param = parameters.get("lambda", 1)
            if lambda_param <= 0 or x < 0:
                return {"error": "Invalid parameter values"}
            pdf = lambda_param * math.exp(-lambda_param * x)

        elif distribution == "uniform":
            a = parameters.get("a", 0)
            b = parameters.get("b", 1)
            if b <= a:
                return {"error": "b must be greater than a"}
            pdf = 1 / (b - a) if a <= x <= b else 0

        else:
            return {"error": f"Unsupported distribution: {distribution}"}

        return {
            "operation": "probability_mass",
            "distribution": distribution,
            "parameters": parameters,
            "x_value": x,
            "probability": pdf,
        }

    def _cumulative_distribution_function(
        self, distribution: str, parameters: Dict[str, float], x: float
    ) -> Dict[str, Any]:
        """Cumulative distribution function"""
        if distribution == "normal":
            mu = parameters.get("mean", parameters.get("mu", 0))
            sigma = parameters.get("std", parameters.get("sigma", 1))
            if sigma <= 0:
                return {"error": "Standard deviation must be positive"}
            # Using error function approximation
            z = (x - mu) / sigma
            cdf = 0.5 * (1 + self._erf(z / math.sqrt(2)))

        elif distribution == "exponential":
            lambda_param = parameters.get("lambda", 1)
            if lambda_param <= 0:
                return {"error": "Lambda must be positive"}
            cdf = 1 - math.exp(-lambda_param * x) if x >= 0 else 0

        elif distribution == "uniform":
            a = parameters.get("a", 0)
            b = parameters.get("b", 1)
            if b <= a:
                return {"error": "b must be greater than a"}
            if x < a:
                cdf = 0
            elif x > b:
                cdf = 1
            else:
                cdf = (x - a) / (b - a)

        else:
            return {
                "error": f"Unsupported distribution for CDF calculation: {distribution}"
            }

        return {
            "operation": "cumulative_distribution",
            "distribution": distribution,
            "parameters": parameters,
            "x_value": x,
            "cumulative_probability": cdf,
        }

    def _inverse_cdf(
        self, distribution: str, parameters: Dict[str, float], p: float
    ) -> Dict[str, Any]:
        """Inverse cumulative distribution function"""
        if not (0 <= p <= 1):
            return {"error": "Probability value must be between [0,1]"}

        if distribution == "exponential":
            lambda_param = parameters.get("lambda", 1)
            if lambda_param <= 0:
                return {"error": "Lambda must be positive"}
            x = -math.log(1 - p) / lambda_param

        elif distribution == "uniform":
            a = parameters.get("a", 0)
            b = parameters.get("b", 1)
            if b <= a:
                return {"error": "b must be greater than a"}
            x = a + p * (b - a)

        elif distribution == "normal":
            mu = parameters.get("mean", parameters.get("mu", 0))
            sigma = parameters.get("std", parameters.get("sigma", 1))
            if sigma <= 0:
                return {"error": "Standard deviation must be positive"}
            x = mu + sigma * self._erf_inverse(2 * p - 1)

        else:
            return {
                "error": f"Unsupported distribution for inverse CDF calculation: {distribution}"
            }

        return {
            "operation": "inverse_cdf",
            "distribution": distribution,
            "parameters": parameters,
            "probability": p,
            "x_value": x,
        }

    def _random_sampling(
        self, distribution: str, parameters: Dict[str, float], n: int
    ) -> Dict[str, Any]:
        """Random sampling"""
        if n <= 0:
            return {"error": "Sample size must be positive"}

        random.seed(42)  # Set seed for reproducibility
        samples = []

        for _ in range(n):
            if distribution == "normal":
                mu = parameters.get("mean", parameters.get("mu", 0))
                sigma = parameters.get("std", parameters.get("sigma", 1))
                samples.append(random.gauss(mu, sigma))

            elif distribution == "uniform":
                a = parameters.get("a", 0)
                b = parameters.get("b", 1)
                samples.append(random.uniform(a, b))

            elif distribution == "exponential":
                lambda_param = parameters.get("lambda", 1)
                samples.append(random.expovariate(lambda_param))

            elif distribution == "binomial":
                n_trials = int(parameters.get("n", 10))
                p = parameters.get("p", 0.5)
                success_count = sum(1 for _ in range(n_trials) if random.random() < p)
                samples.append(success_count)

        # Calculate sample statistics
        mean_sample = sum(samples) / len(samples)
        variance_sample = sum((x - mean_sample) ** 2 for x in samples) / (
            len(samples) - 1
        )
        std_sample = math.sqrt(variance_sample)

        return {
            "operation": "random_sampling",
            "distribution": distribution,
            "parameters": parameters,
            "sample_size": n,
            "samples": samples,
            "sample_statistics": {
                "mean": round(mean_sample, 6),
                "std": round(std_sample, 6),
                "min": min(samples),
                "max": max(samples),
            },
        }

    def _distribution_statistics(
        self, distribution: str, parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """Distribution statistical properties"""
        stats = {}

        if distribution == "normal":
            mu = parameters.get("mean", parameters.get("mu", 0))
            sigma = parameters.get("std", parameters.get("sigma", 1))
            stats = {
                "mean": mu,
                "variance": sigma**2,
                "std": sigma,
                "skewness": 0,
                "kurtosis": 3,
                "mode": mu,
                "median": mu,
            }

        elif distribution == "exponential":
            lambda_param = parameters.get("lambda", 1)
            stats = {
                "mean": 1 / lambda_param,
                "variance": 1 / (lambda_param**2),
                "std": 1 / lambda_param,
                "skewness": 2,
                "kurtosis": 9,
                "mode": 0,
                "median": math.log(2) / lambda_param,
            }

        elif distribution == "uniform":
            a = parameters.get("a", 0)
            b = parameters.get("b", 1)
            stats = {
                "mean": (a + b) / 2,
                "variance": (b - a) ** 2 / 12,
                "std": (b - a) / math.sqrt(12),
                "skewness": 0,
                "kurtosis": 1.8,
                "mode": "any value in [a,b]",
                "median": (a + b) / 2,
            }

        elif distribution == "binomial":
            n = parameters.get("n", 10)
            p = parameters.get("p", 0.5)
            stats = {
                "mean": n * p,
                "variance": n * p * (1 - p),
                "std": math.sqrt(n * p * (1 - p)),
                "skewness": (1 - 2 * p) / math.sqrt(n * p * (1 - p)),
                "mode": math.floor((n + 1) * p),
            }

        return {
            "operation": "distribution_stats",
            "distribution": distribution,
            "parameters": parameters,
            "statistics": stats,
        }

    def _bayes_theorem(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bayes' theorem calculation"""
        if not events or len(events) < 2:
            return {"error": "At least two events are required"}

        # Assume the first event is the one we want to calculate posterior probability for
        # Format: {"name": "A", "prior": 0.3, "likelihood": 0.8}
        target_event = events[0]

        if "prior" not in target_event or "likelihood" not in target_event:
            return {
                "error": "Target event must contain prior probability and likelihood"
            }

        # Calculate total probability
        total_probability = 0
        for event in events:
            if "prior" in event and "likelihood" in event:
                total_probability += event["prior"] * event["likelihood"]

        if total_probability == 0:
            return {"error": "Total probability is zero"}

        # Calculate posterior probability
        posterior = (
            target_event["prior"] * target_event["likelihood"]
        ) / total_probability

        return {
            "operation": "bayes_theorem",
            "target_event": target_event["name"],
            "prior_probability": target_event["prior"],
            "likelihood": target_event["likelihood"],
            "total_probability": total_probability,
            "posterior_probability": round(posterior, 6),
            "events": events,
        }

    def _combinatorics_calculation(
        self, parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combinatorics calculation"""
        operation_type = parameters.get("type", "combination")
        n = int(parameters.get("n", 5))
        r = int(parameters.get("r", 2))

        if n < 0 or r < 0 or r > n:
            return {"error": "Invalid parameter values"}

        if operation_type == "combination":
            result = self._combination(n, r)
            formula = f"C({n},{r}) = {n}!/({r}!({n}-{r})!)"

        elif operation_type == "permutation":
            result = self._permutation(n, r)
            formula = f"P({n},{r}) = {n}!/({n}-{r})!"

        elif operation_type == "factorial":
            result = math.factorial(n)
            formula = f"{n}!"

        else:
            return {"error": f"Unsupported combinatorics operation: {operation_type}"}

        return {
            "operation": "combinatorics",
            "type": operation_type,
            "n": n,
            "r": r,
            "result": result,
            "formula": formula,
        }

    def _hypothesis_testing(
        self, data: List[float], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hypothesis testing"""
        if not data:
            return {"error": "Data cannot be empty"}

        test_type = parameters.get("test_type", "one_sample_t")
        null_hypothesis = parameters.get("null_mean", 0)
        alpha = parameters.get("alpha", 0.05)

        n = len(data)
        sample_mean = sum(data) / n
        sample_std = math.sqrt(sum((x - sample_mean) ** 2 for x in data) / (n - 1))

        if test_type == "one_sample_t":
            # One-sample t-test
            t_statistic = (sample_mean - null_hypothesis) / (sample_std / math.sqrt(n))
            degrees_freedom = n - 1

            # Simplified critical value (two-tailed test)
            critical_value = 2.0  # Approximate value
            p_value = 2 * (1 - self._t_cdf(abs(t_statistic), degrees_freedom))

            reject_null = abs(t_statistic) > critical_value

            return {
                "operation": "hypothesis_test",
                "test_type": test_type,
                "null_hypothesis": f"μ = {null_hypothesis}",
                "sample_size": n,
                "sample_mean": round(sample_mean, 6),
                "sample_std": round(sample_std, 6),
                "t_statistic": round(t_statistic, 6),
                "degrees_of_freedom": degrees_freedom,
                "p_value": round(p_value, 6),
                "alpha": alpha,
                "reject_null": reject_null,
                "conclusion": (
                    "Reject null hypothesis"
                    if reject_null
                    else "Fail to reject null hypothesis"
                ),
            }

        else:
            return {"error": f"Unsupported test type: {test_type}"}

    def _confidence_interval(
        self, data: List[float], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Confidence interval calculation"""
        if not data:
            return {"error": "Data cannot be empty"}

        confidence_level = parameters.get("confidence", 0.95)
        alpha = 1 - confidence_level

        n = len(data)
        sample_mean = sum(data) / n
        sample_std = math.sqrt(sum((x - sample_mean) ** 2 for x in data) / (n - 1))
        standard_error = sample_std / math.sqrt(n)

        # Simplified t-value (95% confidence interval)
        t_value = 2.0  # Approximate value
        margin_error = t_value * standard_error

        lower_bound = sample_mean - margin_error
        upper_bound = sample_mean + margin_error

        return {
            "operation": "confidence_interval",
            "confidence_level": confidence_level,
            "sample_size": n,
            "sample_mean": round(sample_mean, 6),
            "sample_std": round(sample_std, 6),
            "standard_error": round(standard_error, 6),
            "margin_of_error": round(margin_error, 6),
            "confidence_interval": [round(lower_bound, 6), round(upper_bound, 6)],
            "interpretation": f"We are {confidence_level*100}% confident that the population mean is between [{round(lower_bound, 6)}, {round(upper_bound, 6)}]",
        }

    def _monte_carlo_simulation(
        self, parameters: Dict[str, Any], n_simulations: int
    ) -> Dict[str, Any]:
        """Monte Carlo simulation"""
        simulation_type = parameters.get("type", "pi_estimation")

        if simulation_type == "pi_estimation":
            # Estimate π value
            inside_circle = 0
            random.seed(42)

            for _ in range(n_simulations):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                if x**2 + y**2 <= 1:
                    inside_circle += 1

            pi_estimate = 4 * inside_circle / n_simulations
            error = abs(pi_estimate - self.pi)

            return {
                "operation": "monte_carlo",
                "simulation_type": simulation_type,
                "n_simulations": n_simulations,
                "inside_circle": inside_circle,
                "pi_estimate": round(pi_estimate, 6),
                "actual_pi": round(self.pi, 6),
                "error": round(error, 6),
                "error_percentage": round(error / self.pi * 100, 4),
            }

        else:
            return {"error": f"Unsupported simulation type: {simulation_type}"}

    # Helper functions
    def _combination(self, n: int, r: int) -> int:
        """Combination"""
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

    def _permutation(self, n: int, r: int) -> int:
        """Permutation"""
        if r > n or r < 0:
            return 0
        return math.factorial(n) // math.factorial(n - r)

    def _erf(self, x: float) -> float:
        """Error function approximation"""
        # Simplified error function approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
            -x * x
        )

        return sign * y

    def _t_cdf(self, t: float, df: int) -> float:
        """Simplified approximation of t-distribution CDF"""
        # This is a very simplified implementation
        if df > 30:
            # For large degrees of freedom, approximate as standard normal distribution
            return 0.5 * (1 + self._erf(t / math.sqrt(2)))
        else:
            # Simplified calculation
            return 0.5 + 0.5 * math.tanh(t / 2)
