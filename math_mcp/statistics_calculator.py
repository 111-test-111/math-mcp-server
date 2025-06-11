# -*- coding: utf-8 -*-
"""
统计分析模块
提供完整丰富的统计分析功能
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class StatisticsCalculator:
    """统计分析计算器类，提供完整的统计分析功能"""

    def __init__(self):
        """初始化统计分析计算器"""
        pass

    def statistics_analyzer_tool(
        self,
        data1: List[float],
        analysis_type: str,
        data2: Optional[List[float]] = None,
        test_type: Optional[str] = None,
        confidence: float = 0.95,
        distribution_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        综合统计分析工具 - 合并所有统计相关操作

        Args:
            data1: 第一组数据
            analysis_type: 分析类型 ('descriptive', 'tests', 'distribution', 'confidence_interval')
            data2: 第二组数据（某些分析需要）
            test_type: 检验类型（'normality', 'hypothesis', 'correlation'）或分布分析类型（'fitting', 'percentiles', 'outliers'）
            confidence: 置信水平
            distribution_type: 分布类型（用于分布拟合）

        Returns:
            统计分析结果
        """
        try:
            if analysis_type == "descriptive":
                return self.descriptive_statistics(data1)
            elif analysis_type == "tests" and test_type:
                if test_type == "normality":
                    return self.normality_tests(data1)
                elif test_type == "hypothesis":
                    if data2 is not None:
                        return self.hypothesis_testing(data1, data2, "two_sample_t")
                    else:
                        return self.hypothesis_testing(data1, test_type="one_sample_t")
                elif test_type == "correlation":
                    if data2 is None:
                        return {"error": "相关性分析需要两组数据"}
                    return self.correlation_analysis(data1, data2)
                else:
                    return {"error": f"不支持的检验类型: {test_type}"}
            elif analysis_type == "distribution":
                if test_type == "fitting":
                    return self.distribution_fitting(data1)
                elif test_type == "percentiles":
                    return self.percentiles(data1)
                elif test_type == "outliers":
                    return self.outlier_detection(data1)
                else:
                    return {"error": f"不支持的分布分析类型: {test_type}"}
            elif analysis_type == "confidence_interval":
                return self.confidence_interval(data1, confidence)
            else:
                return {"error": f"不支持的分析类型: {analysis_type}"}
        except Exception as e:
            return {"error": f"统计分析出错: {str(e)}"}

    def descriptive_statistics(self, data: List[float]) -> Dict[str, Any]:
        """
        描述性统计分析

        Args:
            data: 数据列表

        Returns:
            描述性统计结果
        """
        try:
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
            return {"error": f"描述性统计计算出错: {str(e)}"}

    def normality_tests(self, data: List[float]) -> Dict[str, Any]:
        """
        正态性检验

        Args:
            data: 数据列表

        Returns:
            正态性检验结果
        """
        try:
            arr = np.array(data)

            # Shapiro-Wilk检验
            shapiro_stat, shapiro_p = stats.shapiro(arr)

            # Jarque-Bera检验
            jarque_stat, jarque_p = stats.jarque_bera(arr)

            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = stats.kstest(
                arr, "norm", args=(np.mean(arr), np.std(arr, ddof=1))
            )

            # Anderson-Darling检验
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
            return {"error": f"正态性检验出错: {str(e)}"}

    def distribution_fitting(self, data: List[float]) -> Dict[str, Any]:
        """
        分布拟合

        Args:
            data: 数据列表

        Returns:
            分布拟合结果
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

                    # 计算AIC和BIC
                    log_likelihood = np.sum(dist.logpdf(arr, *params))
                    k = len(params)  # 参数个数
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
            return {"error": f"分布拟合出错: {str(e)}"}

    def hypothesis_testing(
        self,
        data1: List[float],
        data2: Optional[List[float]] = None,
        test_type: str = "one_sample_t",
        null_hypothesis: float = 0.0,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        假设检验

        Args:
            data1: 第一组数据
            data2: 第二组数据（双样本检验需要）
            test_type: 检验类型
            null_hypothesis: 零假设值
            alpha: 显著性水平

        Returns:
            假设检验结果
        """
        try:
            arr1 = np.array(data1)

            if test_type == "one_sample_t":
                stat, p_value = stats.ttest_1samp(arr1, null_hypothesis)

            elif test_type == "two_sample_t":
                if data2 is None:
                    return {"error": "双样本t检验需要两组数据"}
                arr2 = np.array(data2)
                stat, p_value = stats.ttest_ind(arr1, arr2)

            elif test_type == "paired_t":
                if data2 is None:
                    return {"error": "配对t检验需要两组数据"}
                arr2 = np.array(data2)
                stat, p_value = stats.ttest_rel(arr1, arr2)

            elif test_type == "mann_whitney":
                if data2 is None:
                    return {"error": "Mann-Whitney检验需要两组数据"}
                arr2 = np.array(data2)
                stat, p_value = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")

            elif test_type == "wilcoxon":
                if data2 is None:
                    return {"error": "Wilcoxon检验需要两组数据"}
                arr2 = np.array(data2)
                stat, p_value = stats.wilcoxon(arr1, arr2)

            elif test_type == "levene":
                if data2 is None:
                    return {"error": "Levene检验需要两组数据"}
                arr2 = np.array(data2)
                stat, p_value = stats.levene(arr1, arr2)

            elif test_type == "f_test":
                if data2 is None:
                    return {"error": "F检验需要两组数据"}
                arr2 = np.array(data2)
                stat = np.var(arr1, ddof=1) / np.var(arr2, ddof=1)
                df1, df2 = len(arr1) - 1, len(arr2) - 1
                p_value = 2 * min(
                    stats.f.cdf(stat, df1, df2), 1 - stats.f.cdf(stat, df1, df2)
                )

            else:
                return {"error": f"不支持的检验类型: {test_type}"}

            return {
                "test_type": test_type,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "alpha": alpha,
                "conclusion": "拒绝零假设" if p_value < alpha else "接受零假设",
            }
        except Exception as e:
            return {"error": f"假设检验出错: {str(e)}"}

    def correlation_analysis(
        self, data1: List[float], data2: List[float], method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        相关性分析

        Args:
            data1: 第一组数据
            data2: 第二组数据
            method: 相关性方法 ('pearson', 'spearman', 'kendall')

        Returns:
            相关性分析结果
        """
        try:
            arr1 = np.array(data1)
            arr2 = np.array(data2)

            if len(arr1) != len(arr2):
                return {"error": "两组数据长度必须相同"}

            if method == "pearson":
                corr, p_value = stats.pearsonr(arr1, arr2)
            elif method == "spearman":
                corr, p_value = stats.spearmanr(arr1, arr2)
            elif method == "kendall":
                corr, p_value = stats.kendalltau(arr1, arr2)
            else:
                return {"error": f"不支持的相关性方法: {method}"}

            return {
                "method": method,
                "correlation": float(corr),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
        except Exception as e:
            return {"error": f"相关性分析出错: {str(e)}"}

    def anova_analysis(
        self, groups: List[List[float]], test_type: str = "one_way"
    ) -> Dict[str, Any]:
        """
        方差分析

        Args:
            groups: 各组数据
            test_type: 检验类型 ('one_way', 'two_way')

        Returns:
            方差分析结果
        """
        try:
            if test_type == "one_way":
                stat, p_value = stats.f_oneway(*groups)
                return {
                    "test_type": "one_way_anova",
                    "f_statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            else:
                return {"error": f"暂不支持的方差分析类型: {test_type}"}
        except Exception as e:
            return {"error": f"方差分析出错: {str(e)}"}

    def percentiles(
        self, data: List[float], percentiles: List[float] = None
    ) -> Dict[str, Any]:
        """
        计算分位数

        Args:
            data: 数据列表
            percentiles: 分位数列表（默认[25, 50, 75, 90, 95, 99]）

        Returns:
            分位数结果
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
            return {"error": f"分位数计算出错: {str(e)}"}

    def outlier_detection(
        self, data: List[float], method: str = "iqr"
    ) -> Dict[str, Any]:
        """
        异常值检测

        Args:
            data: 数据列表
            method: 检测方法 ('iqr', 'z_score', 'modified_z_score')

        Returns:
            异常值检测结果
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
                return {"error": f"不支持的异常值检测方法: {method}"}

        except Exception as e:
            return {"error": f"异常值检测出错: {str(e)}"}

    def confidence_interval(
        self, data: List[float], confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        置信区间计算

        Args:
            data: 数据列表
            confidence: 置信水平

        Returns:
            置信区间结果
        """
        try:
            arr = np.array(data)
            n = len(arr)
            mean = np.mean(arr)
            std_err = stats.sem(arr)

            # t分布置信区间
            t_ci = stats.t.interval(confidence, n - 1, loc=mean, scale=std_err)

            return {
                "confidence_level": confidence,
                "mean": float(mean),
                "standard_error": float(std_err),
                "confidence_interval": [float(t_ci[0]), float(t_ci[1])],
                "margin_of_error": float(t_ci[1] - mean),
            }
        except Exception as e:
            return {"error": f"置信区间计算出错: {str(e)}"}
