# -*- coding: utf-8 -*-
"""
Statistical Analysis Module
Provides comprehensive and rich statistical analysis functionality
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Any, Optional, Union, Tuple


class StatisticsCalculator:
    """Statistical analysis calculator class, providing comprehensive statistical analysis functionality"""

    def __init__(self):
        """Initialize statistical analysis calculator"""
        pass

    def statistics_analyzer_tool(
        self,
        data1: List[float],
        analysis_type: str,
        data2: Optional[List[float]] = None,
        test_type: Optional[str] = None,
        hypothesis_test_type: Optional[str] = None,
        confidence: float = 0.95,
        distribution_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis tool - merges all statistics-related operations

        Args:
            data1: First dataset
            analysis_type: Analysis type ('descriptive', 'tests', 'distribution', 'confidence_interval', 'distribution_fit', 'outlier_detection')
            data2: Second dataset (required for certain analyses)
            test_type: Test type ('normality', 'hypothesis', 'correlation') or distribution analysis type ('fitting', 'percentiles', 'outliers')
            hypothesis_test_type: Specific type of hypothesis test (e.g., 'one_sample_t', 'two_sample_t', 'paired_t')
            confidence: Confidence level
            distribution_type: Distribution type (used for distribution fitting)

        Returns:
            Statistical analysis results
        """
        try:
            # # Backward compatibility for legacy calling methods
            # if analysis_type == "distribution_fit":
            #     return self.distribution_fitting(data1)
            # elif analysis_type == "outlier_detection":
            #     return self.outlier_detection(data1)

            if analysis_type == "descriptive":
                return self._descriptive_statistics(data1)
            elif analysis_type == "tests" and test_type:
                if test_type == "normality":
                    return self._normality_tests(data1)
                elif test_type == "hypothesis":
                    ht_type = hypothesis_test_type or (
                        "two_sample_t" if data2 is not None else "one_sample_t"
                    )
                    return self._hypothesis_testing(data1, data2, ht_type)
                elif test_type == "correlation":
                    if data2 is None:
                        return {"error": "Correlation analysis requires two datasets"}
                    return self._correlation_analysis(data1, data2)
                else:
                    return {"error": f"Unsupported test type: {test_type}"}
            elif analysis_type == "distribution":
                if test_type == "fitting":
                    dist_type = distribution_type or "all"
                    return self._distribution_fitting(
                        data1, distributions=[dist_type] if dist_type != "all" else None
                    )
                elif test_type == "percentiles":
                    return self._percentiles(data1)
                elif test_type == "outliers":
                    return self._outlier_detection(data1)
                else:
                    return {
                        "error": f"Unsupported distribution analysis type: {test_type}"
                    }
            elif analysis_type == "confidence_interval":
                return self._confidence_interval(data1, confidence)
            else:
                return {"error": f"Unsupported analysis type: {analysis_type}"}
        except Exception as e:
            return {"error": f"Statistical analysis error: {str(e)}"}

    def _descriptive_statistics(self, data: List[float]) -> Dict[str, Any]:
        """
        Descriptive statistical analysis

        Args:
            data: Data list

        Returns:
            Descriptive statistics results
        """
        try:
            # Check if data is empty
            if not data or len(data) == 0:
                return {"error": "Data list cannot be empty"}

            # Check if data contains valid numeric values
            if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in data):
                return {"error": "Data list must contain valid numeric values"}

            arr = np.array(data)
            return {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)),
                "variance": float(np.var(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "skewness": float(stats.skew(arr)),
                "kurtosis": float(stats.kurtosis(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                "coefficient_of_variation": (
                    float(np.std(arr, ddof=1) / np.mean(arr))
                    if np.mean(arr) != 0
                    else None
                ),
            }
        except Exception as e:
            return {"error": f"Descriptive statistics calculation error: {str(e)}"}

    def _normality_tests(self, data: List[float]) -> Dict[str, Any]:
        """
        Normality tests

        Args:
            data: Data list

        Returns:
            Normality test results
        """
        try:
            arr = np.array(data)

            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(arr)

            # Jarque-Bera test
            jarque_stat, jarque_p = stats.jarque_bera(arr)

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(
                arr, "norm", args=(np.mean(arr), np.std(arr, ddof=1))
            )

            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(arr, dist="norm")

            return {
                "shapiro_test": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05,
                },
                "jarque_bera_test": {
                    "statistic": float(jarque_stat),
                    "p_value": float(jarque_p),
                    "is_normal": jarque_p > 0.05,
                },
                "ks_test": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_p > 0.05,
                },
                "anderson_darling_test": {
                    "statistic": float(ad_stat),
                    "critical_values": ad_critical.tolist(),
                    "significance_levels": ad_significance.tolist(),
                },
            }
        except Exception as e:
            return {"error": f"Normality test error: {str(e)}"}

    def _distribution_fitting(self, data: List[float]) -> Dict[str, Any]:
        """
        Distribution fitting

        Args:
            data: Data list

        Returns:
            Distribution fitting results
        """
        try:
            arr = np.array(data)
            distributions = [
                "norm",
                "uniform",
                "expon",
                "gamma",
                "beta",
                "lognorm",
                "chi2",
                "t",
            ]
            results = {}

            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(arr)
                    ks_stat, ks_p = stats.kstest(arr, lambda x: dist.cdf(x, *params))

                    # Calculate AIC and BIC
                    log_likelihood = np.sum(dist.logpdf(arr, *params))
                    k = len(params)  # Number of parameters
                    n = len(arr)
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood

                    results[dist_name] = {
                        "parameters": params,
                        "ks_test_statistic": float(ks_stat),
                        "ks_test_p_value": float(ks_p),
                        "fits_well": ks_p > 0.05,
                        "aic": float(aic),
                        "bic": float(bic),
                        "log_likelihood": float(log_likelihood),
                    }
                except:
                    continue

            return results
        except Exception as e:
            return {"error": f"Distribution fitting error: {str(e)}"}

    def _hypothesis_testing(
        self,
        data1: List[float],
        data2: Optional[List[float]] = None,
        test_type: str = "one_sample_t",
        null_hypothesis: float = 0.0,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Hypothesis testing

        Args:
            data1: First dataset
            data2: Second dataset (required for two-sample tests)
            test_type: Test type
            null_hypothesis: Null hypothesis value
            alpha: Significance level

        Returns:
            Hypothesis test results
        """
        try:
            arr1 = np.array(data1)

            if test_type == "one_sample_t":
                stat, p_value = stats.ttest_1samp(arr1, null_hypothesis)

            elif test_type == "two_sample_t":
                if data2 is None:
                    return {"error": "Two-sample t-test requires two datasets"}
                arr2 = np.array(data2)
                stat, p_value = stats.ttest_ind(arr1, arr2)

            elif test_type == "paired_t":
                if data2 is None:
                    return {"error": "Paired t-test requires two datasets"}
                arr2 = np.array(data2)
                stat, p_value = stats.ttest_rel(arr1, arr2)

            elif test_type == "wilcoxon":
                if data2 is None:
                    return {"error": "Wilcoxon test requires two datasets"}
                arr2 = np.array(data2)
                stat, p_value = stats.wilcoxon(arr1, arr2)

            elif test_type in ["mann_whitney_u", "mannwhitneyu", "ttest_ind"]:
                if data2 is None:
                    return {"error": "Mann-Whitney U test requires two datasets"}
                arr2 = np.array(data2)
                if test_type == "ttest_ind":
                    # Alias for independent samples t-test
                    stat, p_value = stats.ttest_ind(arr1, arr2)
                else:
                    stat, p_value = stats.mannwhitneyu(
                        arr1, arr2, alternative="two-sided"
                    )

            elif test_type == "levene":
                if data2 is None:
                    return {"error": "Levene test requires two datasets"}
                arr2 = np.array(data2)
                stat, p_value = stats.levene(arr1, arr2)

            elif test_type == "f_test":
                if data2 is None:
                    return {"error": "F-test requires two datasets"}
                arr2 = np.array(data2)
                stat = np.var(arr1, ddof=1) / np.var(arr2, ddof=1)
                df1, df2 = len(arr1) - 1, len(arr2) - 1
                p_value = 2 * min(
                    stats.f.cdf(stat, df1, df2), 1 - stats.f.cdf(stat, df1, df2)
                )

            else:
                return {"error": f"Unsupported test type: {test_type}"}

            return {
                "test_type": test_type,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "alpha": alpha,
                "conclusion": (
                    "Reject null hypothesis"
                    if p_value < alpha
                    else "Accept null hypothesis"
                ),
            }
        except Exception as e:
            return {"error": f"Hypothesis testing error: {str(e)}"}

    def _correlation_analysis(
        self, data1: List[float], data2: List[float], method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Correlation analysis

        Args:
            data1: First dataset
            data2: Second dataset
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation analysis results
        """
        try:
            arr1 = np.array(data1)
            arr2 = np.array(data2)

            if len(arr1) != len(arr2):
                return {"error": "Both datasets must have the same length"}

            if method == "pearson":
                corr, p_value = stats.pearsonr(arr1, arr2)
            elif method == "spearman":
                corr, p_value = stats.spearmanr(arr1, arr2)
            elif method == "kendall":
                corr, p_value = stats.kendalltau(arr1, arr2)
            else:
                return {"error": f"Unsupported correlation method: {method}"}

            return {
                "method": method,
                "correlation": float(corr),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
        except Exception as e:
            return {"error": f"Correlation analysis error: {str(e)}"}

    def _anova_analysis(
        self, groups: List[List[float]], test_type: str = "one_way"
    ) -> Dict[str, Any]:
        """
        Analysis of Variance (ANOVA)

        Args:
            groups: List of data groups
            test_type: Test type ('one_way')

        Returns:
            ANOVA test results
        """
        try:
            if test_type == "one_way":
                if len(groups) < 2:
                    return {
                        "error": "One-way ANOVA requires at least two groups of data"
                    }

                f_stat, p_value = stats.f_oneway(*groups)

                return {
                    "test_type": "one_way_anova",
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "conclusion": (
                        "Reject null hypothesis, significant differences between group means"
                        if p_value < 0.05
                        else "Cannot reject null hypothesis, no significant differences between group means"
                    ),
                }
            # two_way ANOVA has been removed due to complex data input requirements.
            # It requires data in a pandas DataFrame format with columns for the value, factor1, and factor2.
            # To re-enable, a more complex data input structure would be needed for this tool.
            else:
                return {"error": f"Unsupported ANOVA test type: {test_type}"}

        except Exception as e:
            return {"error": f"ANOVA analysis error: {str(e)}"}

    def _percentiles(
        self, data: List[float], percentiles: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate percentiles

        Args:
            data: Data list
            percentiles: List of percentiles (default [25, 50, 75, 90, 95, 99])

        Returns:
            Percentile results
        """
        try:
            arr = np.array(data)
            if percentiles is None:
                percentiles = [25, 50, 75, 90, 95, 99]

            results = {}
            for p in percentiles:
                results[f"p{p}"] = float(np.percentile(arr, p))

            return results
        except Exception as e:
            return {"error": f"Percentile calculation error: {str(e)}"}

    def _outlier_detection(
        self, data: List[float], method: str = "iqr"
    ) -> Dict[str, Any]:
        """
        Outlier detection

        Args:
            data: Data list
            method: Detection method ('iqr', 'z_score', 'modified_z_score')

        Returns:
            Outlier detection results
        """
        try:
            arr = np.array(data)

            if method == "iqr":
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = arr[(arr < lower_bound) | (arr > upper_bound)]

                return {
                    "method": "iqr",
                    "outliers": outliers.tolist(),
                    "outlier_count": len(outliers),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

            elif method == "z_score":
                z_scores = np.abs(stats.zscore(arr))
                outliers = arr[z_scores > 3]

                return {
                    "method": "z_score",
                    "outliers": outliers.tolist(),
                    "outlier_count": len(outliers),
                    "threshold": 3.0,
                }

            elif method == "modified_z_score":
                median = np.median(arr)
                mad = np.median(np.abs(arr - median))
                modified_z_scores = 0.6745 * (arr - median) / mad
                outliers = arr[np.abs(modified_z_scores) > 3.5]

                return {
                    "method": "modified_z_score",
                    "outliers": outliers.tolist(),
                    "outlier_count": len(outliers),
                    "threshold": 3.5,
                }

            else:
                return {"error": f"Unsupported outlier detection method: {method}"}

        except Exception as e:
            return {"error": f"Outlier detection error: {str(e)}"}

    def _confidence_interval(
        self, data: List[float], confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Confidence interval calculation

        Args:
            data: Data list
            confidence: Confidence level

        Returns:
            Confidence interval results
        """
        try:
            arr = np.array(data)
            n = len(arr)
            mean = np.mean(arr)
            std_err = stats.sem(arr)

            # t-distribution confidence interval
            t_ci = stats.t.interval(confidence, n - 1, loc=mean, scale=std_err)

            return {
                "confidence_level": confidence,
                "mean": float(mean),
                "standard_error": float(std_err),
                "confidence_interval": [float(t_ci[0]), float(t_ci[1])],
                "margin_of_error": float(t_ci[1] - mean),
            }
        except Exception as e:
            return {"error": f"Confidence interval calculation error: {str(e)}"}
