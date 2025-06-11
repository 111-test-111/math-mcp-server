# -*- coding: utf-8 -*-
"""
回归分析模块
提供完整丰富的回归分析功能
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")


class RegressionCalculator:
    """回归分析计算器类，提供完整的回归分析功能"""

    def __init__(self):
        """初始化回归分析计算器"""
        pass

    def regression_modeler_tool(
        self,
        x_data: List[List[float]],
        y_data: List[float],
        model_type: str = "linear",
        degree: int = 2,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        cv_folds: int = 5,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        综合回归建模工具

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据
            model_type: 模型类型 ('linear', 'polynomial', 'ridge', 'lasso', 'elastic_net', 'logistic')
            degree: 多项式次数（用于多项式回归）
            alpha: 正则化参数（用于正则化回归）
            l1_ratio: L1正则化比例（用于ElasticNet）
            cv_folds: 交叉验证折数
            test_size: 测试集比例

        Returns:
            回归建模结果
        """
        try:
            if model_type == "linear":
                return self.linear_regression(x_data, y_data)
            elif model_type == "polynomial":
                if len(x_data[0]) > 1:
                    return {"error": "多项式回归只支持单变量输入"}
                x_1d = [row[0] for row in x_data]
                return self.polynomial_regression(x_1d, y_data, degree)
            elif model_type == "ridge":
                return self.ridge_regression(x_data, y_data, alpha)
            elif model_type == "lasso":
                return self.lasso_regression(x_data, y_data, alpha)
            elif model_type == "elastic_net":
                return self.elastic_net_regression(x_data, y_data, alpha, l1_ratio)
            elif model_type == "logistic":
                y_int = [int(y) for y in y_data]
                return self.logistic_regression(x_data, y_int)
            else:
                return {"error": f"不支持的模型类型: {model_type}"}
        except Exception as e:
            return {"error": f"回归建模出错: {str(e)}"}

    def linear_regression(
        self, x_data: List[List[float]], y_data: List[float]
    ) -> Dict[str, Any]:
        """
        线性回归分析

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据

        Returns:
            线性回归结果
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # 拟合线性回归模型
            model = LinearRegression()
            model.fit(X, y)

            # 预测
            y_pred = model.predict(X)

            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 调整R²
            n = len(y)
            p = X.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            return {
                "model_type": "linear_regression",
                "coefficients": {
                    "intercept": float(model.intercept_),
                    "slopes": model.coef_.tolist(),
                },
                "statistics": {
                    "r_squared": float(r2),
                    "adjusted_r_squared": float(adjusted_r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                },
                "predictions": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }

        except Exception as e:
            return {"error": f"线性回归分析出错: {str(e)}"}

    def polynomial_regression(
        self, x_data: List[float], y_data: List[float], degree: int = 2
    ) -> Dict[str, Any]:
        """
        多项式回归分析

        Args:
            x_data: 自变量数据（一维）
            y_data: 因变量数据
            degree: 多项式次数

        Returns:
            多项式回归结果
        """
        try:
            X = np.array(x_data).reshape(-1, 1)
            y = np.array(y_data)

            # 生成多项式特征
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            # 拟合多项式回归模型
            model = LinearRegression()
            model.fit(X_poly, y)

            # 预测
            y_pred = model.predict(X_poly)

            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            return {
                "model_type": "polynomial_regression",
                "degree": degree,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    "coefficients": model.coef_[1:].tolist(),
                },
                "statistics": {
                    "r_squared": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                },
                "predictions": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }

        except Exception as e:
            return {"error": f"多项式回归分析出错: {str(e)}"}

    def ridge_regression(
        self, x_data: List[List[float]], y_data: List[float], alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        岭回归分析

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据
            alpha: 正则化参数

        Returns:
            岭回归结果
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 拟合岭回归模型
            model = Ridge(alpha=alpha)
            model.fit(X_scaled, y)

            # 预测
            y_pred = model.predict(X_scaled)

            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 交叉验证评分
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

            return {
                "model_type": "ridge_regression",
                "alpha": alpha,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    "slopes": model.coef_.tolist(),
                },
                "statistics": {
                    "r_squared": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "cv_r2_mean": float(cv_scores.mean()),
                    "cv_r2_std": float(cv_scores.std()),
                },
                "predictions": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }

        except Exception as e:
            return {"error": f"岭回归分析出错: {str(e)}"}

    def lasso_regression(
        self, x_data: List[List[float]], y_data: List[float], alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Lasso回归分析

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据
            alpha: 正则化参数

        Returns:
            Lasso回归结果
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 拟合Lasso回归模型
            model = Lasso(alpha=alpha, max_iter=2000)
            model.fit(X_scaled, y)

            # 预测
            y_pred = model.predict(X_scaled)

            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 特征选择信息
            selected_features = np.where(model.coef_ != 0)[0].tolist()
            n_selected = len(selected_features)

            # 交叉验证评分
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

            return {
                "model_type": "lasso_regression",
                "alpha": alpha,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    "slopes": model.coef_.tolist(),
                },
                "feature_selection": {
                    "selected_features": selected_features,
                    "n_selected": n_selected,
                    "n_total": X.shape[1],
                },
                "statistics": {
                    "r_squared": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "cv_r2_mean": float(cv_scores.mean()),
                    "cv_r2_std": float(cv_scores.std()),
                },
                "predictions": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }

        except Exception as e:
            return {"error": f"Lasso回归分析出错: {str(e)}"}

    def elastic_net_regression(
        self,
        x_data: List[List[float]],
        y_data: List[float],
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        弹性网络回归分析

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据
            alpha: 正则化强度
            l1_ratio: L1正则化比例

        Returns:
            弹性网络回归结果
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 拟合弹性网络回归模型
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
            model.fit(X_scaled, y)

            # 预测
            y_pred = model.predict(X_scaled)

            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 特征选择信息
            selected_features = np.where(model.coef_ != 0)[0].tolist()
            n_selected = len(selected_features)

            # 交叉验证评分
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

            return {
                "model_type": "elastic_net_regression",
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    "slopes": model.coef_.tolist(),
                },
                "feature_selection": {
                    "selected_features": selected_features,
                    "n_selected": n_selected,
                    "n_total": X.shape[1],
                },
                "statistics": {
                    "r_squared": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "cv_r2_mean": float(cv_scores.mean()),
                    "cv_r2_std": float(cv_scores.std()),
                },
                "predictions": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }

        except Exception as e:
            return {"error": f"弹性网络回归分析出错: {str(e)}"}

    def logistic_regression(
        self, x_data: List[List[float]], y_data: List[int]
    ) -> Dict[str, Any]:
        """
        逻辑回归分析

        Args:
            x_data: 自变量数据矩阵
            y_data: 因变量数据（二分类：0或1）

        Returns:
            逻辑回归结果
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 拟合逻辑回归模型
            model = LogisticRegression(max_iter=2000)
            model.fit(X_scaled, y)

            # 预测
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]

            # 计算评估指标
            accuracy = model.score(X_scaled, y)

            # 混淆矩阵
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y, y_pred)

            # 精确度、召回率、F1分数
            from sklearn.metrics import precision_score, recall_score, f1_score

            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)

            # AUC-ROC
            from sklearn.metrics import roc_auc_score

            auc_roc = roc_auc_score(y, y_pred_proba)

            return {
                "model_type": "logistic_regression",
                "coefficients": {
                    "intercept": float(model.intercept_[0]),
                    "slopes": model.coef_[0].tolist(),
                },
                "statistics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "auc_roc": float(auc_roc),
                },
                "confusion_matrix": cm.tolist(),
                "predictions": {
                    "classes": y_pred.tolist(),
                    "probabilities": y_pred_proba.tolist(),
                },
            }

        except Exception as e:
            return {"error": f"逻辑回归分析出错: {str(e)}"}

    def residual_analysis(
        self, y_true: List[float], y_pred: List[float]
    ) -> Dict[str, Any]:
        """
        残差分析

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            残差分析结果
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            residuals = y_true - y_pred

            # 残差统计
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q50": float(np.percentile(residuals, 50)),
                "q75": float(np.percentile(residuals, 75)),
            }

            # 正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # 异方差性检验（Breusch-Pagan test的简化版本）
            # 计算残差平方与预测值的相关性
            squared_residuals = residuals**2
            correlation_coef = np.corrcoef(squared_residuals, y_pred)[0, 1]

            # 杜宾-沃森统计量（检验自相关性）
            durbin_watson = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)

            return {
                "residual_statistics": residual_stats,
                "normality_test": {
                    "shapiro_statistic": float(shapiro_stat),
                    "shapiro_p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05,
                },
                "heteroscedasticity": {
                    "residuals_vs_fitted_correlation": float(correlation_coef),
                    "potential_heteroscedasticity": abs(correlation_coef) > 0.3,
                },
                "autocorrelation": {
                    "durbin_watson_statistic": float(durbin_watson),
                    "interpretation": (
                        "positive_autocorr"
                        if durbin_watson < 1.5
                        else (
                            "negative_autocorr"
                            if durbin_watson > 2.5
                            else "no_autocorr"
                        )
                    ),
                },
                "residuals": residuals.tolist(),
            }

        except Exception as e:
            return {"error": f"残差分析出错: {str(e)}"}

    def model_comparison(self, models_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        模型比较

        Args:
            models_results: 多个模型的结果列表

        Returns:
            模型比较结果
        """
        try:
            comparison = []

            for i, model_result in enumerate(models_results):
                if "statistics" in model_result:
                    stats = model_result["statistics"]
                    comparison.append(
                        {
                            "model_index": i,
                            "model_type": model_result.get("model_type", f"model_{i}"),
                            "r_squared": stats.get("r_squared", None),
                            "adjusted_r_squared": stats.get("adjusted_r_squared", None),
                            "mse": stats.get("mse", None),
                            "rmse": stats.get("rmse", None),
                            "mae": stats.get("mae", None),
                            "cv_r2_mean": stats.get("cv_r2_mean", None),
                        }
                    )

            # 找到最佳模型
            best_r2_idx = max(
                range(len(comparison)),
                key=lambda i: comparison[i]["r_squared"] or -float("inf"),
            )
            best_rmse_idx = min(
                range(len(comparison)),
                key=lambda i: comparison[i]["rmse"] or float("inf"),
            )

            return {
                "comparison_table": comparison,
                "best_models": {
                    "highest_r_squared": {
                        "index": best_r2_idx,
                        "model_type": comparison[best_r2_idx]["model_type"],
                        "r_squared": comparison[best_r2_idx]["r_squared"],
                    },
                    "lowest_rmse": {
                        "index": best_rmse_idx,
                        "model_type": comparison[best_rmse_idx]["model_type"],
                        "rmse": comparison[best_rmse_idx]["rmse"],
                    },
                },
            }

        except Exception as e:
            return {"error": f"模型比较出错: {str(e)}"}
