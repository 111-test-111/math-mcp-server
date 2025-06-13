# -*- coding: utf-8 -*-
"""
Regression Analysis Module
Provides comprehensive and rich regression analysis functionality
"""

import numpy as np
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


class RegressionCalculator:
    """Regression analysis calculator class, providing comprehensive regression analysis functionality"""

    def __init__(self):
        """Initialize regression analysis calculator"""
        pass

    def regression_modeler_tool(
        self,
        operation: str = "fit",
        x_data: Optional[List[List[float]]] = None,
        y_data: Optional[List[float]] = None,
        model_type: str = "linear",
        degree: int = 2,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        cv_folds: int = 5,
        test_size: float = 0.2,
        y_true: Optional[List[float]] = None,
        y_pred: Optional[List[float]] = None,
        models_results: Optional[List[Dict[str, Any]]] = None,
        training_x: Optional[List[List[float]]] = None,
        training_y: Optional[List[float]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive regression modeling tool

        Args:
            operation: Operation type ('fit', 'residual_analysis', 'model_comparison')
            x_data: Independent variable data matrix (for 'fit')
            y_data: Dependent variable data (for 'fit')
            model_type: Model type ('linear', 'polynomial', 'ridge', 'lasso', 'elastic_net', 'logistic')
            degree: Polynomial degree (for polynomial regression)
            alpha: Regularization parameter (for regularized regression)
            l1_ratio: L1 regularization ratio (for ElasticNet)
            cv_folds: Cross-validation folds
            test_size: Test set ratio
            y_true: True values (for 'residual_analysis')
            y_pred: Predicted values (for 'residual_analysis')
            models_results: Model results list (for 'model_comparison')

        Returns:
            Regression modeling or analysis results
        """
        try:
            if operation == "fit":
                if x_data is None or y_data is None:
                    return {"error": "'fit' operation requires x_data and y_data"}
                if model_type == "linear":
                    return self._linear_regression(x_data, y_data)
                elif model_type == "polynomial":
                    if len(x_data[0]) > 1:
                        return {
                            "error": "Polynomial regression only supports single variable input"
                        }
                    x_1d = [row[0] for row in x_data]
                    return self._polynomial_regression(x_1d, y_data, degree)
                elif model_type == "ridge":
                    return self._ridge_regression(x_data, y_data, alpha)
                elif model_type == "lasso":
                    return self._lasso_regression(x_data, y_data, alpha)
                elif model_type == "elastic_net":
                    return self._elastic_net_regression(x_data, y_data, alpha, l1_ratio)
                elif model_type == "logistic":
                    y_int = [int(y) for y in y_data]
                    return self._logistic_regression(x_data, y_int)
                else:
                    return {"error": f"Unsupported model type: {model_type}"}

            elif operation == "predict":
                if x_data is None:
                    return {"error": "'predict' operation requires x_data"}
                return self._predict_with_model(
                    x_data,
                    model_type,
                    degree,
                    alpha,
                    l1_ratio,
                    training_x,
                    training_y,
                    model_params,
                )

            elif operation == "residual_analysis":
                if y_true is None or y_pred is None:
                    return {
                        "error": "'residual_analysis' operation requires y_true and y_pred"
                    }
                return self._residual_analysis(y_true, y_pred)

            elif operation == "model_comparison":
                if models_results is None:
                    return {
                        "error": "'model_comparison' operation requires models_results"
                    }
                return self._model_comparison(models_results)

            else:
                return {"error": f"Unsupported operation type: {operation}"}
        except Exception as e:
            return {"error": f"Regression modeling error: {str(e)}"}

    def _linear_regression(
        self, x_data: List[List[float]], y_data: List[float]
    ) -> Dict[str, Any]:
        """
        Linear regression analysis

        Args:
            x_data: Independent variable data matrix
            y_data: Dependent variable data

        Returns:
            Linear regression results
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Prediction
            y_pred = model.predict(X)

            # Calculate evaluation metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Adjusted R²
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
            return {"error": f"Linear regression analysis error: {str(e)}"}

    def _polynomial_regression(
        self, x_data: List[float], y_data: List[float], degree: int = 2
    ) -> Dict[str, Any]:
        """
        Polynomial regression analysis

        Args:
            x_data: Independent variable data (one-dimensional)
            y_data: Dependent variable data
            degree: Polynomial degree

        Returns:
            Polynomial regression results
        """
        try:
            X = np.array(x_data).reshape(-1, 1)
            y = np.array(y_data)

            # Generate polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            # Fit polynomial regression model
            model = LinearRegression()
            model.fit(X_poly, y)

            # Prediction
            y_pred = model.predict(X_poly)

            # Calculate evaluation metrics
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
            return {"error": f"Polynomial regression analysis error: {str(e)}"}

    def _ridge_regression(
        self, x_data: List[List[float]], y_data: List[float], alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Ridge regression analysis

        Args:
            x_data: Independent variable data matrix
            y_data: Dependent variable data
            alpha: Regularization parameter

        Returns:
            Ridge regression results
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit ridge regression model
            model = Ridge(alpha=alpha)
            model.fit(X_scaled, y)

            # Prediction
            y_pred = model.predict(X_scaled)

            # Calculate evaluation metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Cross-validation score
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
            return {"error": f"Ridge regression analysis error: {str(e)}"}

    def _lasso_regression(
        self, x_data: List[List[float]], y_data: List[float], alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Lasso regression analysis

        Args:
            x_data: Independent variable data matrix
            y_data: Dependent variable data
            alpha: Regularization parameter

        Returns:
            Lasso regression results
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit Lasso regression model
            model = Lasso(alpha=alpha, max_iter=2000)
            model.fit(X_scaled, y)

            # Prediction
            y_pred = model.predict(X_scaled)

            # Calculate evaluation metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Feature selection information
            selected_features = np.where(model.coef_ != 0)[0].tolist()
            n_selected = len(selected_features)

            # Cross-validation score
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
            return {"error": f"Lasso regression analysis error: {str(e)}"}

    def _elastic_net_regression(
        self,
        x_data: List[List[float]],
        y_data: List[float],
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Elastic net regression analysis

        Args:
            x_data: Independent variable data matrix
            y_data: Dependent variable data
            alpha: Regularization strength
            l1_ratio: L1 regularization ratio

        Returns:
            Elastic net regression results
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit elastic net regression model
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
            model.fit(X_scaled, y)

            # Prediction
            y_pred = model.predict(X_scaled)

            # Calculate evaluation metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Feature selection information
            selected_features = np.where(model.coef_ != 0)[0].tolist()
            n_selected = len(selected_features)

            # Cross-validation score
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
            return {"error": f"Elastic net regression analysis error: {str(e)}"}

    def _logistic_regression(
        self, x_data: List[List[float]], y_data: List[int]
    ) -> Dict[str, Any]:
        """
        Logistic regression analysis

        Args:
            x_data: Independent variable data matrix
            y_data: Dependent variable data (binary classification: 0 or 1)

        Returns:
            Logistic regression results
        """
        try:
            X = np.array(x_data)
            y = np.array(y_data)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit logistic regression model
            model = LogisticRegression(max_iter=2000)
            model.fit(X_scaled, y)

            # Prediction
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]

            # Calculate evaluation metrics
            accuracy = model.score(X_scaled, y)

            # Confusion matrix
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y, y_pred)

            # Precision, recall, F1 score
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
            return {"error": f"Logistic regression analysis error: {str(e)}"}

    def _residual_analysis(
        self, y_true: List[float], y_pred: List[float]
    ) -> Dict[str, Any]:
        """
        Residual analysis

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Residual analysis results
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            residuals = y_true - y_pred

            # Residual statistics
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q50": float(np.percentile(residuals, 50)),
                "q75": float(np.percentile(residuals, 75)),
            }

            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Heteroscedasticity test (simplified version of Breusch-Pagan test)
            # Calculate correlation between squared residuals and predicted values
            squared_residuals = residuals**2
            correlation_coef = np.corrcoef(squared_residuals, y_pred)[0, 1]

            # Durbin-Watson statistic (test for autocorrelation)
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
            return {"error": f"Residual analysis error: {str(e)}"}

    def _model_comparison(self, models_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Model comparison

        Args:
            models_results: List of multiple model results

        Returns:
            Model comparison results
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

            # Find best model
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
            return {"error": f"Model comparison error: {str(e)}"}

    def _predict_with_model(
        self,
        x_data: List[List[float]],
        model_type: str = "linear",
        degree: int = 2,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        training_x: Optional[List[List[float]]] = None,
        training_y: Optional[List[float]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make predictions using trained model

        Args:
            x_data: Independent variable data for prediction
            model_type: Model type
            degree: Polynomial degree
            alpha: Regularization parameter
            l1_ratio: L1 regularization ratio
            training_x: Training data X (if model parameters not provided)
            training_y: Training data y (if model parameters not provided)
            model_params: Pre-trained model parameters

        Returns:
            Prediction results
        """
        try:
            X_pred = np.array(x_data)

            if model_params is not None:
                # Use provided model parameters for prediction
                if model_type == "linear":
                    intercept = model_params.get("intercept", 0)
                    slopes = model_params.get("slopes", [])

                    if len(slopes) != X_pred.shape[1]:
                        return {
                            "error": f"Feature count mismatch: expected {len(slopes)} features, got {X_pred.shape[1]}"
                        }

                    predictions = []
                    for row in X_pred:
                        pred = intercept + sum(
                            slope * feature for slope, feature in zip(slopes, row)
                        )
                        predictions.append(pred)

                    return {
                        "model_type": model_type,
                        "predictions": predictions,
                        "input_data": x_data,
                    }

                elif model_type == "polynomial":
                    # For polynomial regression, need to reconstruct features
                    if X_pred.shape[1] != 1:
                        return {
                            "error": "Polynomial regression only supports single variable input"
                        }

                    coefficients = model_params.get("coefficients", [])
                    predictions = []

                    for row in X_pred:
                        x_val = row[0]
                        pred = sum(
                            coef * (x_val**i) for i, coef in enumerate(coefficients)
                        )
                        predictions.append(pred)

                    return {
                        "model_type": model_type,
                        "degree": degree,
                        "predictions": predictions,
                        "input_data": x_data,
                    }

            else:
                # If no model parameters provided, need training data to retrain
                if training_x is None or training_y is None:
                    return {
                        "error": "Need to provide model_params or training_x and training_y"
                    }

                # Retrain model and predict
                if model_type == "linear":
                    X_train = np.array(training_x)
                    y_train = np.array(training_y)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_pred)

                    return {
                        "model_type": model_type,
                        "predictions": predictions.tolist(),
                        "input_data": x_data,
                        "model_coefficients": {
                            "intercept": float(model.intercept_),
                            "slopes": model.coef_.tolist(),
                        },
                    }

                elif model_type == "polynomial":
                    if X_pred.shape[1] != 1:
                        return {
                            "error": "Polynomial regression only supports single variable input"
                        }

                    X_train = np.array(training_x)
                    y_train = np.array(training_y)

                    # Generate polynomial features
                    poly = PolynomialFeatures(degree=degree)
                    X_train_poly = poly.fit_transform(X_train)
                    X_pred_poly = poly.transform(X_pred)

                    model = LinearRegression()
                    model.fit(X_train_poly, y_train)
                    predictions = model.predict(X_pred_poly)

                    return {
                        "model_type": model_type,
                        "degree": degree,
                        "predictions": predictions.tolist(),
                        "input_data": x_data,
                        "model_coefficients": {
                            "intercept": float(model.intercept_),
                            "coefficients": model.coef_.tolist(),
                        },
                    }

                elif model_type == "ridge":
                    X_train = np.array(training_x)
                    y_train = np.array(training_y)

                    model = Ridge(alpha=alpha)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_pred)

                    return {
                        "model_type": model_type,
                        "predictions": predictions.tolist(),
                        "input_data": x_data,
                        "model_coefficients": {
                            "intercept": float(model.intercept_),
                            "slopes": model.coef_.tolist(),
                        },
                    }

                else:
                    return {
                        "error": f"Prediction functionality not yet supported for model type: {model_type}"
                    }

        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
