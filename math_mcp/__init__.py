# -*- coding: utf-8 -*-
"""
数学计算MCP服务器包
提供强大的数学计算工具，包括基础数值计算、矩阵计算、统计分析、微积分等
"""

__version__ = "1.1.0"
__author__ = "111-test-111"

# 导出主要模块
from .basic_calculator import BasicCalculator
from .matrix_calculator import MatrixCalculator
from .statistics_calculator import StatisticsCalculator
from .calculus_calculator import CalculusCalculator
from .optimization_calculator import OptimizationCalculator
from .regression_calculator import RegressionCalculator
from .plotting_calculator import PlottingCalculator

__all__ = [
    "BasicCalculator",
    "MatrixCalculator",
    "StatisticsCalculator",
    "CalculusCalculator",
    "OptimizationCalculator",
    "RegressionCalculator",
    "PlottingCalculator",
]
