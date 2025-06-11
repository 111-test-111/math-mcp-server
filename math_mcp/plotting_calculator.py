# -*- coding: utf-8 -*-
"""
统计绘图模块
提供完整丰富的统计图表绘制功能，返回base64编码图片供LLM使用
"""

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.font_manager as fm
import os


def setup_font():
    """设置语言性字体支持"""

    # 读取环境变量FONT_PATH
    font_path = os.getenv("FONT_PATH")
    if font_path and os.path.exists(font_path):
        # 添加字体到管理器，确保可用
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        # 使用系统默认中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 初始化字体设置
setup_font()


class PlottingCalculator:
    """统计绘图计算器类，提供完整的统计图表绘制功能"""

    def __init__(self):
        """初始化绘图计算器"""

        # 设置默认样式
        sns.set_style("whitegrid")
        setup_font()
        self.default_figsize = (10, 6)
        self.default_dpi = 300
        self.default_colors = sns.color_palette("husl", 10)

    def statistical_plotter_tool(
        self,
        chart_type: str,
        data: Optional[List[float]] = None,
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
        y_data_series: Optional[List[List[float]]] = None,
        series_labels: Optional[List[str]] = None,
        matrix_data: Optional[List[List[float]]] = None,
        labels: Optional[List[str]] = None,
        title: str = "统计图表",
        xlabel: str = "X轴",
        ylabel: str = "Y轴",
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
        style: str = "whitegrid",
        show_values: bool = False,
        horizontal: bool = False,
        trend_line: bool = False,
        trend_line_color: Optional[str] = None,
        trend_line_equation: Optional[str] = None,
        bins: int = 30,
        annotate: bool = True,
        colormap: str = "viridis",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        综合统计绘图工具 - 支持多种统计图表类型

        Args:
            chart_type: 图表类型 ('bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap', 'correlation_matrix', 'multi_series_line')
            data: 单组数据（用于柱状图、饼图、直方图、箱线图）
            x_data: X轴数据（用于线图、散点图、多系列线图）
            y_data: Y轴数据（用于线图、散点图）
            y_data_series: 多系列Y轴数据（用于多系列线图）
            series_labels: 多系列图的标签
            matrix_data: 矩阵数据（用于热力图、多组箱线图、相关性矩阵）
            labels: 数据标签
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            colors: 颜色列表
            figsize: 图片大小 (width, height)
            dpi: 图片分辨率
            style: 图表样式
            show_values: 是否显示数值（柱状图）
            horizontal: 是否为水平图表（柱状图）
            trend_line: 是否显示趋势线（散点图）
            trend_line_color: 趋势线颜色
            trend_line_equation: 趋势线方程
            bins: 直方图分箱数量
            annotate: 是否显示标注（热力图）
            colormap: 颜色映射（热力图）
            **kwargs: 其他图表参数

        Returns:
            包含base64编码图片的结果字典
        """
        try:
            # 设置默认图片大小
            actual_figsize = figsize
            if actual_figsize is None:
                if chart_type == "pie":
                    actual_figsize = (8, 8)
                elif chart_type == "heatmap":
                    actual_figsize = (10, 8)
                else:
                    actual_figsize = (10, 6)

            # 基础通用参数
            base_kwargs = {
                "figsize": actual_figsize,
                "dpi": dpi,
                "style": style,
                "colors": colors,
                "xlabel": xlabel,
                "ylabel": ylabel,
            }

            if chart_type == "bar" and data:
                bar_kwargs = {
                    **base_kwargs,
                    "show_values": show_values,
                    "horizontal": horizontal,
                }
                return self.bar_chart(data, labels, title, **bar_kwargs)

            elif chart_type == "pie" and data:
                return self.pie_chart(data, labels, title, **base_kwargs)

            elif chart_type == "line" and x_data and y_data:
                line_kwargs = {
                    **base_kwargs,
                    "color": kwargs.get("color", "blue"),
                    "line_width": kwargs.get("line_width", 2.0),
                    "line_style": kwargs.get("line_style", "-"),
                    "marker": kwargs.get("marker", "o"),
                    "marker_size": kwargs.get("marker_size", 6),
                    "alpha": kwargs.get("alpha", 1.0),
                }
                return self.line_chart(x_data, y_data, title, **line_kwargs)

            elif chart_type == "scatter" and x_data and y_data:
                scatter_kwargs = {
                    **base_kwargs,
                    "color": kwargs.get("color", "blue"),
                    "marker_size": kwargs.get("marker_size", 6),
                    "alpha": kwargs.get("alpha", 0.7),
                    "trend_line": trend_line,
                    "trend_line_color": trend_line_color,
                    "trend_line_equation": trend_line_equation,
                }
                return self.scatter_plot(x_data, y_data, title, **scatter_kwargs)

            elif chart_type == "histogram" and data:
                hist_kwargs = {
                    **base_kwargs,
                    "bins": bins,
                    "color": kwargs.get("color", "skyblue"),
                }
                return self.histogram(data, title, **hist_kwargs)

            elif chart_type == "box":
                if data:
                    return self.box_plot(data, title, **base_kwargs)
                elif matrix_data:
                    return self.box_plot(matrix_data, title, **base_kwargs)
                else:
                    return {"error": "箱线图需要提供data或matrix_data参数"}

            elif chart_type == "heatmap" and matrix_data:
                heatmap_kwargs = {
                    **base_kwargs,
                    "colormap": colormap,
                    "annotate": annotate,
                }
                return self.heatmap(matrix_data, title, **heatmap_kwargs)

            elif chart_type == "correlation_matrix" and matrix_data:
                corr_kwargs = {
                    "labels": labels,
                    "title": title,
                    "figsize": actual_figsize,
                    "dpi": dpi,
                    "style": style,
                    "colormap": colormap,
                    "annotate": annotate,
                }
                return self.correlation_matrix(matrix_data, **corr_kwargs)

            elif chart_type == "multi_series_line" and x_data and y_data_series:
                multi_line_kwargs = {
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "series_labels": series_labels,
                    "colors": colors,
                    "line_styles": kwargs.get("line_styles"),
                    "markers": kwargs.get("markers"),
                    "figsize": actual_figsize,
                    "dpi": dpi,
                    "style": style,
                    "grid": kwargs.get("grid", True),
                    "legend": kwargs.get("legend", True),
                }
                return self.multi_series_line_chart(
                    x_data, y_data_series, **multi_line_kwargs
                )

            else:
                return {"error": f"不支持的图表类型: {chart_type} 或缺少必要的数据参数"}

        except Exception as e:
            return {"error": f"统计绘图出错: {str(e)}"}

    def _prepare_figure(
        self, figsize: Tuple[float, float], dpi: int, style: str
    ) -> plt.Figure:
        """准备图形对象"""
        if style:
            sns.set_style(style)
            setup_font()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig, ax

    def _save_to_base64(self, fig: plt.Figure, format: str = "png") -> str:
        """将图形保存为base64编码字符串"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=format, bbox_inches="tight", facecolor="white")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return image_base64
        finally:
            # 确保资源总是被清理
            try:
                buffer.close()
            except:
                pass
            try:
                plt.close(fig)
            except:
                pass
            # 强制垃圾回收
            import gc

            gc.collect()

    def _apply_styling(
        self,
        ax,
        title: str,
        xlabel: str,
        ylabel: str,
        title_fontsize: int,
        label_fontsize: int,
        tick_fontsize: int,
        grid: bool,
        legend: bool,
    ):
        """应用通用样式设置"""
        if title:
            ax.set_title(title, fontsize=title_fontsize, pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)

        ax.tick_params(labelsize=tick_fontsize)

        if not grid:
            ax.grid(False)

        if legend and ax.get_legend():
            ax.legend(fontsize=label_fontsize)

    def bar_chart(
        self,
        data: List[float],
        labels: Optional[List[str]] = None,
        title: str = "柱状图",
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制柱状图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            colors = kwargs.get("colors", self.default_colors[: len(data)])

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if labels is None:
                labels = [f"类别{i+1}" for i in range(len(data))]

            bars = ax.bar(labels, data, color=colors)
            ax.set_title(title, fontsize=16)

            if kwargs.get("show_values", True):
                for bar, value in zip(bars, data):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(data) * 0.01,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                    )

            plt.xticks(rotation=45 if len(labels) > 5 else 0)
            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "bar_chart",
                "image_base64": image_base64,
                "data_summary": {"categories": len(data), "total": sum(data)},
            }
        except Exception as e:
            return {"error": f"柱状图绘制出错: {str(e)}"}

    def pie_chart(
        self,
        data: List[float],
        labels: Optional[List[str]] = None,
        title: str = "饼图",
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制饼图"""
        try:
            figsize = kwargs.get("figsize", (8, 8))
            dpi = kwargs.get("dpi", 300)
            colors = kwargs.get("colors", self.default_colors[: len(data)])

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if labels is None:
                labels = [f"类别{i+1}" for i in range(len(data))]

            ax.pie(data, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title(title, fontsize=16)

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "pie_chart",
                "image_base64": image_base64,
                "data_summary": {"categories": len(data), "total": sum(data)},
            }
        except Exception as e:
            return {"error": f"饼图绘制出错: {str(e)}"}

    def line_chart(
        self, x_data: List[float], y_data: List[float], title: str = "线图", **kwargs
    ) -> Dict[str, Any]:
        """绘制线图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            color = kwargs.get("color", "blue")
            linewidth = kwargs.get("line_width", 2.0)
            line_style = kwargs.get("line_style", "-")
            marker = kwargs.get("marker", "o")
            marker_size = kwargs.get("marker_size", 6)
            alpha = kwargs.get("alpha", 1.0)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax.plot(
                x_data,
                y_data,
                color=color,
                linestyle=line_style,
                marker=marker,
                linewidth=linewidth,
                markersize=marker_size,
                alpha=alpha,
            )
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(kwargs.get("xlabel", "X轴"))
            ax.set_ylabel(kwargs.get("ylabel", "Y轴"))

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "line_chart",
                "image_base64": image_base64,
                "data_summary": {"points": len(x_data)},
            }
        except Exception as e:
            return {"error": f"线图绘制出错: {str(e)}"}

    def scatter_plot(
        self, x_data: List[float], y_data: List[float], title: str = "散点图", **kwargs
    ) -> Dict[str, Any]:
        """绘制散点图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            color = kwargs.get("color", "blue")
            marker_size = kwargs.get("marker_size", 6)
            alpha = kwargs.get("alpha", 0.7)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax.scatter(x_data, y_data, color=color, s=marker_size * 10, alpha=alpha)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(kwargs.get("xlabel", "X轴"))
            ax.set_ylabel(kwargs.get("ylabel", "Y轴"))

            # 添加趋势线（如果指定）
            if kwargs.get("trend_line", False):
                try:
                    import numpy as np
                    from sklearn.linear_model import LinearRegression

                    X = np.array(x_data).reshape(-1, 1)
                    y = np.array(y_data)
                    model = LinearRegression().fit(X, y)
                    trend_y = model.predict(X)

                    trend_color = kwargs.get("trend_line_color", "red")
                    ax.plot(
                        x_data, trend_y, color=trend_color, linestyle="--", linewidth=2
                    )

                    # 添加方程（如果提供）
                    if kwargs.get("trend_line_equation"):
                        ax.text(
                            0.05,
                            0.95,
                            kwargs["trend_line_equation"],
                            transform=ax.transAxes,
                            fontsize=10,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        )
                except Exception as e:
                    print(f"趋势线绘制失败: {str(e)}")

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "scatter_plot",
                "image_base64": image_base64,
                "data_summary": {"points": len(x_data)},
            }
        except Exception as e:
            return {"error": f"散点图绘制出错: {str(e)}"}

    def histogram(
        self, data: List[float], title: str = "直方图", **kwargs
    ) -> Dict[str, Any]:
        """绘制直方图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            bins = kwargs.get("bins", 30)
            color = kwargs.get("color", "skyblue")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            n, bins_edges, patches = ax.hist(data, bins=bins, color=color, alpha=0.7)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(kwargs.get("xlabel", "数值"))
            ax.set_ylabel(kwargs.get("ylabel", "频次"))

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "histogram",
                "image_base64": image_base64,
                "data_summary": {"bins": len(bins_edges) - 1, "total_count": len(data)},
            }
        except Exception as e:
            return {"error": f"直方图绘制出错: {str(e)}"}

    def box_plot(
        self,
        data: Union[List[float], List[List[float]]],
        title: str = "箱线图",
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制箱线图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if isinstance(data[0], list):
                # 多组数据
                ax.boxplot(data)
                ax.set_title(title, fontsize=16)
                ax.set_xlabel(kwargs.get("xlabel", "组别"))
                ax.set_ylabel(kwargs.get("ylabel", "数值"))
                data_summary = {"groups": len(data)}
            else:
                # 单组数据
                ax.boxplot(data)
                ax.set_title(title, fontsize=16)
                ax.set_ylabel(kwargs.get("ylabel", "数值"))
                data_summary = {"values": len(data)}

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "box_plot",
                "image_base64": image_base64,
                "data_summary": data_summary,
            }
        except Exception as e:
            return {"error": f"箱线图绘制出错: {str(e)}"}

    def heatmap(
        self, data: List[List[float]], title: str = "热力图", **kwargs
    ) -> Dict[str, Any]:
        """绘制热力图"""
        try:
            figsize = kwargs.get("figsize", (10, 8))
            dpi = kwargs.get("dpi", 300)
            colormap = kwargs.get("colormap", "viridis")
            annotate = kwargs.get("annotate", True)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            import numpy as np

            data_array = np.array(data)
            im = ax.imshow(data_array, cmap=colormap, aspect="auto")

            # 添加颜色条
            plt.colorbar(im, ax=ax)

            # 添加数值标注
            if annotate:
                for i in range(data_array.shape[0]):
                    for j in range(data_array.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{data_array[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if data_array[i, j] < data_array.mean()
                                else "black"
                            ),
                        )

            ax.set_title(title, fontsize=16)

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "heatmap",
                "image_base64": image_base64,
                "data_summary": {"shape": data_array.shape},
            }
        except Exception as e:
            return {"error": f"热力图绘制出错: {str(e)}"}

    def correlation_matrix(
        self,
        data: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "相关性矩阵",
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 300,
        style: str = "whitegrid",
        title_fontsize: int = 16,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        colormap: str = "RdBu_r",
        annotate: bool = True,
        fmt: str = ".2f",
    ) -> Dict[str, Any]:
        """绘制相关性矩阵热力图"""
        try:
            import pandas as pd
            import numpy as np

            # 转换为DataFrame便于计算相关性
            if labels is None:
                labels = [f"变量{i+1}" for i in range(len(data[0]))]

            df = pd.DataFrame(data, columns=labels)
            correlation_matrix = df.corr()

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # 绘制热力图
            im = ax.imshow(
                correlation_matrix.values, cmap=colormap, vmin=-1, vmax=1, aspect="auto"
            )

            # 设置刻度和标签
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=tick_fontsize)
            ax.set_yticklabels(labels, fontsize=tick_fontsize)

            # 旋转x轴标签
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # 添加数值标注
            if annotate:
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        value = correlation_matrix.iloc[i, j]
                        color = "white" if abs(value) > 0.6 else "black"
                        ax.text(
                            j,
                            i,
                            f"{value:{fmt}}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=10,
                        )

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("相关系数", fontsize=label_fontsize)

            # 设置标题
            ax.set_title(title, fontsize=title_fontsize, pad=20)

            # 调整布局
            plt.tight_layout()

            # 保存为base64
            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "correlation_matrix",
                "image_base64": image_base64,
                "correlation_data": correlation_matrix.to_dict(),
                "data_summary": {
                    "variables": len(labels),
                    "observations": len(data),
                    "strong_correlations": int(
                        np.sum(np.abs(correlation_matrix.values) > 0.7) - len(labels)
                    ),
                },
            }

        except Exception as e:
            return {"error": f"相关性矩阵绘制出错: {str(e)}"}

    def multi_series_line_chart(
        self,
        x_data: List[float],
        y_data_series: List[List[float]],
        series_labels: Optional[List[str]] = None,
        title: str = "多系列线图",
        xlabel: str = "X轴",
        ylabel: str = "Y轴",
        colors: Optional[List[str]] = None,
        line_styles: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 300,
        style: str = "whitegrid",
        title_fontsize: int = 16,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        grid: bool = True,
        legend: bool = True,
    ) -> Dict[str, Any]:
        """绘制多系列线图"""
        try:
            # 设置样式
            if style:
                sns.set_style(style)
                setup_font()

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # 设置默认参数
            num_series = len(y_data_series)
            if series_labels is None:
                series_labels = [f"系列{i+1}" for i in range(num_series)]

            if colors is None:
                colors = self.default_colors[:num_series]

            if line_styles is None:
                line_styles = ["-"] * num_series

            if markers is None:
                markers = ["o", "s", "^", "D", "v"] * (num_series // 5 + 1)

            # 绘制每个系列
            for i, (y_data, label, color, line_style, marker) in enumerate(
                zip(
                    y_data_series,
                    series_labels,
                    colors,
                    line_styles,
                    markers[:num_series],
                )
            ):
                ax.plot(
                    x_data,
                    y_data,
                    label=label,
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    markersize=4,
                    linewidth=2,
                    alpha=0.8,
                )

            # 应用样式
            self._apply_styling(
                ax,
                title,
                xlabel,
                ylabel,
                title_fontsize,
                label_fontsize,
                tick_fontsize,
                grid,
                legend,
            )

            # 保存为base64
            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "multi_series_line_chart",
                "image_base64": image_base64,
                "data_summary": {
                    "series_count": num_series,
                    "data_points_per_series": len(x_data),
                    "series_labels": series_labels,
                },
            }

        except Exception as e:
            return {"error": f"多系列线图绘制出错: {str(e)}"}

    def plot_function_tool(
        self,
        function_expression: str,
        variable: str = "x",
        x_range: Tuple[float, float] = (-10, 10),
        num_points: int = 1000,
        title: str = "函数图像",
        xlabel: str = "X轴",
        ylabel: str = "Y轴",
        filename: Optional[str] = None,
        format: str = "png",
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 300,
        color: str = "blue",
        line_width: float = 2.0,
        grid: bool = True,
        grid_alpha: float = 0.3,
        derivative_order: Optional[int] = None,
        show_critical_points: bool = False,
        show_equation: bool = True,
        equation_position: str = "upper right",
        alpha: float = 1.0,
        line_style: str = "-",
        marker: str = "",
        marker_size: int = 6,
    ) -> Dict[str, Any]:
        """绘制数学函数曲线"""
        try:
            import numpy as np
            import sympy as sp
            from datetime import datetime
            import os
            import pathlib

            # 生成x数据
            x_vals = np.linspace(x_range[0], x_range[1], num_points)

            # 解析函数表达式
            var = sp.Symbol(variable)
            expr = sp.sympify(function_expression)

            # 转换为数值函数
            func = sp.lambdify(var, expr, "numpy")

            # 计算y值
            try:
                y_vals = func(x_vals)
                # 处理可能的复数结果
                if np.iscomplexobj(y_vals):
                    y_vals = np.real(y_vals)
            except Exception as e:
                return {"error": f"函数计算出错: {str(e)}"}

            # 创建图表
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # 绘制主函数
            line_label = (
                f"f({variable}) = {function_expression}" if show_equation else None
            )
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=alpha,
                label=line_label,
                marker=marker if marker else None,
                markersize=marker_size,
                markevery=max(1, num_points // 50) if marker else None,
            )

            # 绘制导数（如果指定）
            if derivative_order is not None:
                try:
                    derivative_expr = sp.diff(expr, var, derivative_order)
                    derivative_func = sp.lambdify(var, derivative_expr, "numpy")
                    derivative_y_vals = derivative_func(x_vals)

                    if np.iscomplexobj(derivative_y_vals):
                        derivative_y_vals = np.real(derivative_y_vals)

                    derivative_color = "red" if color == "blue" else "blue"
                    derivative_label = (
                        f"f{'′' * derivative_order}({variable})"
                        if show_equation
                        else f"{derivative_order}阶导数"
                    )

                    ax.plot(
                        x_vals,
                        derivative_y_vals,
                        color=derivative_color,
                        linewidth=line_width * 0.8,
                        linestyle="--",
                        alpha=alpha * 0.8,
                        label=derivative_label,
                    )
                except Exception as e:
                    print(f"导数绘制警告: {str(e)}")

            # 标记临界点（如果指定）
            if show_critical_points:
                try:
                    first_derivative = sp.diff(expr, var)
                    critical_points = sp.solve(first_derivative, var)

                    for point in critical_points:
                        if point.is_real:
                            point_val = float(point.evalf())
                            if x_range[0] <= point_val <= x_range[1]:
                                y_val = float(expr.subs(var, point).evalf())
                                ax.plot(
                                    point_val,
                                    y_val,
                                    "ro",
                                    markersize=8,
                                    markerfacecolor="red",
                                    markeredgecolor="darkred",
                                    markeredgewidth=2,
                                    label=(
                                        "临界点" if point == critical_points[0] else ""
                                    ),
                                )
                except Exception as e:
                    print(f"临界点计算警告: {str(e)}")

            # 设置标题和标签
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            # 设置网格
            if grid:
                ax.grid(True, alpha=grid_alpha)

            # 显示图例
            if show_equation or derivative_order is not None or show_critical_points:
                ax.legend(loc=equation_position, fontsize=10)

            # 自动调整y轴范围（排除异常值）
            if len(y_vals) > 0 and np.isfinite(y_vals).any():
                finite_y = y_vals[np.isfinite(y_vals)]
                if len(finite_y) > 0:
                    y_mean = np.mean(finite_y)
                    y_std = np.std(finite_y)
                    y_min = max(np.min(finite_y), y_mean - 3 * y_std)
                    y_max = min(np.max(finite_y), y_mean + 3 * y_std)
                    if y_min != y_max:
                        ax.set_ylim(y_min, y_max)

            # 保存图像文件
            # 使用环境变量 OUTPUT_PATH
            output_path = os.getenv("OUTPUT_PATH", "plots")
            output_dir = pathlib.Path(output_path)
            output_dir.mkdir(exist_ok=True, parents=True)

            # 生成文件名
            if not filename:
                filename = f"function_curve_{function_expression.replace('*', 'x').replace('/', 'div')}"

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = filename.replace(f".{format}", "")  # 移除可能存在的扩展名
            full_filename = f"{base_filename}_{timestamp_str}.{format}"
            full_path = output_dir / full_filename

            try:
                fig.savefig(
                    full_path,
                    format=format,
                    bbox_inches="tight",
                    facecolor="white",
                    dpi=dpi,
                )
                saved_path = str(full_path.absolute())
                save_success = True
            except Exception as e:
                saved_path = f"保存失败: {str(e)}"
                save_success = False

            return {
                "success": save_success,
                "chart_type": "function_curve",
                "function": function_expression,
                "variable": variable,
                "x_range": list(x_range),
                "saved_path": saved_path,
                "filename": full_filename if save_success else None,
                "message": (
                    f"函数图像已保存到: {saved_path}" if save_success else saved_path
                ),
                "data_summary": {
                    "x_range": list(x_range),
                    "num_points": num_points,
                    "has_derivative": derivative_order is not None,
                    "has_critical_points": show_critical_points,
                },
            }

        except Exception as e:
            return {"error": f"函数绘图出错: {str(e)}"}
