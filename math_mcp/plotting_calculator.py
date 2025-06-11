# -*- coding: utf-8 -*-
"""
统计绘图模块
提供完整丰富的统计图表绘制功能，返回base64编码图片供LLM使用
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import base64
import io
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


def setup_chinese_font():
    """设置中文字体支持"""
    font_path = os.getenv("FONT_PATH")

    if font_path and os.path.exists(font_path):
        # 使用指定的字体文件
        try:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
            print(f"使用字体: {font_prop.get_name()} ({font_path})")
        except Exception as e:
            print(f"无法加载指定字体 {font_path}: {str(e)}")
            # 回退到默认中文字体
            plt.rcParams["font.sans-serif"] = [
                "SimHei",
                "Microsoft YaHei UI",
                "SimSun",
                "DengXian",
            ]
    else:
        # 使用默认中文字体
        plt.rcParams["font.sans-serif"] = [
            "SimHei",
            "Microsoft YaHei",
            "DejaVu Sans",
            "Arial Unicode MS",
            "sans-serif",
        ]

    plt.rcParams["axes.unicode_minus"] = False


# 初始化字体设置
setup_chinese_font()


class PlottingCalculator:
    """统计绘图计算器类，提供完整的统计图表绘制功能"""

    def __init__(self):
        """初始化绘图计算器"""
        # 设置默认样式
        sns.set_style("whitegrid")
        self.default_figsize = (10, 6)
        self.default_dpi = 300
        self.default_colors = sns.color_palette("husl", 10)

    def statistical_plotter_tool(
        self,
        chart_type: str,
        data: Optional[List[float]] = None,
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
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
            chart_type: 图表类型 ('bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap')
            data: 单组数据（用于柱状图、饼图、直方图、箱线图）
            x_data: X轴数据（用于线图、散点图）
            y_data: Y轴数据（用于线图、散点图）
            matrix_data: 矩阵数据（用于热力图、多组箱线图）
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

            # 构建参数字典
            plot_kwargs = {
                "figsize": actual_figsize,
                "dpi": dpi,
                "style": style,
                "colors": colors,
                "show_values": show_values,
                "horizontal": horizontal,
                "trend_line": trend_line,
                "trend_line_color": trend_line_color,
                "trend_line_equation": trend_line_equation,
                "bins": bins,
                "annotate": annotate,
                "colormap": colormap,
                "xlabel": xlabel,
                "ylabel": ylabel,
                **kwargs,
            }

            if chart_type == "bar" and data:
                return self.bar_chart(data, labels, title, **plot_kwargs)

            elif chart_type == "pie" and data:
                return self.pie_chart(data, labels, title, **plot_kwargs)

            elif chart_type == "line" and x_data and y_data:
                return self.line_chart(x_data, y_data, title, **plot_kwargs)

            elif chart_type == "scatter" and x_data and y_data:
                return self.scatter_plot(x_data, y_data, title, **plot_kwargs)

            elif chart_type == "histogram" and data:
                return self.histogram(data, title, **plot_kwargs)

            elif chart_type == "box":
                if data:
                    return self.box_plot(data, title, **plot_kwargs)
                elif matrix_data:
                    return self.box_plot(matrix_data, title, **plot_kwargs)
                else:
                    return {"error": "箱线图需要提供data或matrix_data参数"}

            elif chart_type == "heatmap" and matrix_data:
                return self.heatmap(matrix_data, title, **plot_kwargs)

            else:
                return {"error": f"不支持的图表类型: {chart_type} 或缺少必要的数据参数"}

        except Exception as e:
            return {"error": f"统计绘图出错: {str(e)}"}

    def _prepare_figure(
        self, figsize: Tuple[float, float], dpi: int, style: str
    ) -> plt.Figure:
        """准备图形对象"""
        # 确保使用正确的字体设置
        setup_chinese_font()

        if style:
            sns.set_style(style)
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

            # 添加趋势线
            if kwargs.get("trend_line", False):
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                trend_color = kwargs.get("trend_line_color", "red")
                ax.plot(x_data, p(x_data), color=trend_color, linestyle="--", alpha=0.8)

            image_base64 = self._save_to_base64(fig)

            correlation = np.corrcoef(x_data, y_data)[0, 1]

            return {
                "chart_type": "scatter_plot",
                "image_base64": image_base64,
                "data_summary": {
                    "points": len(x_data),
                    "correlation": float(correlation),
                },
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

            ax.hist(data, bins=bins, color=color, alpha=0.7)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("数值")
            ax.set_ylabel("频次")

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "histogram",
                "image_base64": image_base64,
                "data_summary": {
                    "count": len(data),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                },
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

            if isinstance(data[0], (int, float)):
                plot_data = [data]
                labels = ["数据"]
            else:
                plot_data = data
                labels = kwargs.get("labels", [f"组{i+1}" for i in range(len(data))])

            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            ax.set_title(title, fontsize=16)

            # 着色
            colors = kwargs.get("colors", self.default_colors[: len(plot_data)])
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "box_plot",
                "image_base64": image_base64,
                "data_summary": {"groups": len(plot_data)},
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

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            data_array = np.array(data)
            im = ax.imshow(data_array, cmap=kwargs.get("colormap", "viridis"))
            ax.set_title(title, fontsize=16)

            # 添加颜色条
            plt.colorbar(im, ax=ax)

            # 添加数值标注
            if kwargs.get("annotate", True):
                for i in range(len(data)):
                    for j in range(len(data[0])):
                        ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center")

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
        """
        绘制相关性矩阵热力图

        Args:
            data: 二维数据矩阵（每列为一个变量）
            labels: 变量标签
            title: 图标题
            figsize: 图片大小
            dpi: 图片分辨率
            style: 图表样式
            title_fontsize: 标题字体大小
            label_fontsize: 标签字体大小
            tick_fontsize: 刻度字体大小
            colormap: 颜色映射
            annotate: 是否显示数值
            fmt: 数值格式

        Returns:
            包含base64图片的结果字典
        """
        try:
            fig, ax = self._prepare_figure(figsize, dpi, style)

            # 转换为DataFrame并计算相关性矩阵
            if labels is None:
                labels = [f"变量{i+1}" for i in range(len(data[0]))]

            df = pd.DataFrame(data, columns=labels)
            corr_matrix = df.corr()

            # 创建掩码以只显示下三角
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=annotate,
                fmt=fmt,
                cmap=colormap,
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": tick_fontsize},
            )

            ax.set_title(title, fontsize=title_fontsize, pad=20)
            ax.tick_params(labelsize=tick_fontsize)

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "correlation_matrix",
                "image_base64": image_base64,
                "image_format": "png",
                "correlation_matrix": corr_matrix.values.tolist(),
                "variable_names": labels,
                "data_summary": {
                    "variables": len(labels),
                    "observations": len(data),
                    "highest_correlation": float(
                        corr_matrix.values[corr_matrix.values < 1].max()
                    ),
                    "lowest_correlation": float(corr_matrix.values.min()),
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
        """
        绘制多系列线图

        Args:
            x_data: X轴数据
            y_data_series: 多个Y轴数据系列
            series_labels: 系列标签
            title: 图标题
            xlabel: X轴标签
            ylabel: Y轴标签
            colors: 颜色列表
            line_styles: 线条样式列表
            markers: 标记样式列表
            figsize: 图片大小
            dpi: 图片分辨率
            style: 图表样式
            title_fontsize: 标题字体大小
            label_fontsize: 标签字体大小
            tick_fontsize: 刻度字体大小
            grid: 是否显示网格
            legend: 是否显示图例

        Returns:
            包含base64图片的结果字典
        """
        try:
            fig, ax = self._prepare_figure(figsize, dpi, style)

            n_series = len(y_data_series)

            if series_labels is None:
                series_labels = [f"系列{i+1}" for i in range(n_series)]

            if colors is None:
                colors = self.default_colors[:n_series]

            if line_styles is None:
                line_styles = ["-"] * n_series

            if markers is None:
                markers = ["o", "s", "^", "v", "D", "*", "+", "x"][:n_series]

            for i, (y_data, label, color, style_line, marker) in enumerate(
                zip(y_data_series, series_labels, colors, line_styles, markers)
            ):
                ax.plot(
                    x_data,
                    y_data,
                    color=color,
                    linestyle=style_line,
                    marker=marker,
                    label=label,
                    linewidth=2,
                    markersize=6,
                )

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

            image_base64 = self._save_to_base64(fig)

            return {
                "chart_type": "multi_series_line_chart",
                "image_base64": image_base64,
                "image_format": "png",
                "data_summary": {
                    "series_count": n_series,
                    "points_per_series": len(x_data),
                    "x_range": [min(x_data), max(x_data)],
                },
            }

        except Exception as e:
            return {"error": f"多系列线图绘制出错: {str(e)}"}
