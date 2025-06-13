# -*- coding: utf-8 -*-
"""
Math MCP Server Package
Provides powerful math calculation tools, including basic numerical calculation, matrix calculation, statistical analysis, calculus, etc.
"""

__version__ = "1.2.0"
__author__ = "111-test-111"

# Export main modules
from .basic import BasicCalculator
from .matrix import MatrixCalculator
from .mstatistics import StatisticsCalculator
from .calculus import CalculusCalculator
from .optimization import OptimizationCalculator
from .regression import RegressionCalculator
from .plotting import PlottingCalculator
from .geometry import GeometryCalculator
from .number_theory import NumberTheoryCalculator
from .signal_processing import SignalProcessingCalculator
from .financial import FinancialCalculator
from .probability import ProbabilityCalculator
from .complex_analysis import ComplexAnalysisCalculator
from .graph_theory import GraphTheoryCalculator

__all__ = [
    "BasicCalculator",
    "MatrixCalculator",
    "StatisticsCalculator",
    "CalculusCalculator",
    "OptimizationCalculator",
    "RegressionCalculator",
    "PlottingCalculator",
    "GeometryCalculator",
    "NumberTheoryCalculator",
    "SignalProcessingCalculator",
    "FinancialCalculator",
    "ProbabilityCalculator",
    "ComplexAnalysisCalculator",
    "GraphTheoryCalculator",
]
