# -*- coding: utf-8 -*-
"""
统计绘图模块
提供完整丰富的统计图表绘制功能，直接保存图片文件到指定路径
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.font_manager as fm
import os
import logging

try:
    from .file_utils import generate_unique_filename
except ImportError:
    from math_mcp.file_utils import generate_unique_filename


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
        filename: Optional[str] = None,
        format: str = "png",
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
        grid: bool = True,
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
            filename: 保存图表的文件名（可选）
            format: 图表保存的格式（png, jpg, svg等）
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
            grid: 是否显示网格
            **kwargs: 其他图表参数

        Returns:
            包含图片保存信息的结果字典
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
                "filename": filename,
                "format": format,
                "grid": grid,
            }

            if chart_type == "bar" and data:
                bar_kwargs = {
                    **base_kwargs,
                    "show_values": show_values,
                    "horizontal": horizontal,
                }
                return self._bar_chart(data, labels, title, **bar_kwargs)

            elif chart_type == "pie" and data:
                return self._pie_chart(data, labels, title, **base_kwargs)

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
                return self._line_chart(x_data, y_data, title, **line_kwargs)

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
                return self._scatter_plot(x_data, y_data, title, **scatter_kwargs)

            elif chart_type == "histogram" and data:
                hist_kwargs = {
                    **base_kwargs,
                    "bins": bins,
                    "color": kwargs.get("color", "skyblue"),
                }
                return self._histogram(data, title, **hist_kwargs)

            elif chart_type == "box":
                if data:
                    return self._box_plot(data, title, **base_kwargs)
                elif matrix_data:
                    return self._box_plot(matrix_data, title, **base_kwargs)
                else:
                    return {"error": "箱线图需要提供data或matrix_data参数"}

            elif chart_type == "heatmap" and matrix_data:
                heatmap_kwargs = {
                    **base_kwargs,
                    "colormap": colormap,
                    "annotate": annotate,
                }
                return self._heatmap(matrix_data, title, **heatmap_kwargs)

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
                return self._correlation_matrix(matrix_data, **corr_kwargs)

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
                return self._multi_series_line_chart(
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

    def _save_figure(
        self,
        fig: plt.Figure,
        format: str = "png",
        custom_filename: Optional[str] = None,
    ) -> Optional[str]:
        """将图形保存到文件

        Args:
            fig: Matplotlib图形对象
            format: 图像格式（png, jpg, svg等）
            custom_filename: 自定义文件名（可选，如不提供则生成默认文件名）

        Returns:
            str: 文件路径（保存成功时）或 None（保存失败时）
        """
        try:
            # 如果没有提供自定义文件名，生成默认文件名
            if custom_filename is None:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_filename = f"chart_{timestamp}"

            # 使用工具函数生成唯一文件名
            file_path, filename = generate_unique_filename(
                "chart", format, custom_filename
            )

            # 保存图像到文件
            try:
                fig.savefig(
                    file_path,
                    format=format,
                    bbox_inches="tight",
                    facecolor="white",
                    dpi=fig.dpi,
                )
                return file_path
            except Exception as e:
                logging.error(f"保存图像到文件时出错: {str(e)}")
                return None

        finally:
            # 确保资源总是被清理
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

    def _bar_chart(
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
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

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
            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "bar_chart",
                "data_summary": {"categories": len(data), "total": sum(data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"柱状图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"柱状图绘制出错: {str(e)}"}

    def _pie_chart(
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
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if labels is None:
                labels = [f"类别{i+1}" for i in range(len(data))]

            ax.pie(data, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title(title, fontsize=16)

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "pie_chart",
                "data_summary": {"categories": len(data), "total": sum(data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"饼图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"饼图绘制出错: {str(e)}"}

    def _line_chart(
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
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

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

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "line_chart",
                "data_summary": {"points": len(x_data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"线图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"线图绘制出错: {str(e)}"}

    def _scatter_plot(
        self, x_data: List[float], y_data: List[float], title: str = "散点图", **kwargs
    ) -> Dict[str, Any]:
        """绘制散点图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            color = kwargs.get("color", "blue")
            marker_size = kwargs.get("marker_size", 6)
            alpha = kwargs.get("alpha", 0.7)
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

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

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "scatter_plot",
                "data_summary": {"points": len(x_data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"散点图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"散点图绘制出错: {str(e)}"}

    def _histogram(
        self, data: List[float], title: str = "直方图", **kwargs
    ) -> Dict[str, Any]:
        """绘制直方图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            bins = kwargs.get("bins", 30)
            color = kwargs.get("color", "skyblue")
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            n, bins_edges, patches = ax.hist(data, bins=bins, color=color, alpha=0.7)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(kwargs.get("xlabel", "数值"))
            ax.set_ylabel(kwargs.get("ylabel", "频次"))

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "histogram",
                "data_summary": {"bins": len(bins_edges) - 1, "total_count": len(data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"直方图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"直方图绘制出错: {str(e)}"}

    def _box_plot(
        self,
        data: Union[List[float], List[List[float]]],
        title: str = "箱线图",
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制箱线图"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

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

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "box_plot",
                "data_summary": data_summary,
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"箱线图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"箱线图绘制出错: {str(e)}"}

    def _heatmap(
        self, data: List[List[float]], title: str = "热力图", **kwargs
    ) -> Dict[str, Any]:
        """绘制热力图"""
        try:
            figsize = kwargs.get("figsize", (10, 8))
            dpi = kwargs.get("dpi", 300)
            colormap = kwargs.get("colormap", "viridis")
            annotate = kwargs.get("annotate", True)
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

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

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "heatmap",
                "data_summary": {"shape": data_array.shape},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = f"热力图已保存到文件: {os.path.basename(file_path)}"
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"热力图绘制出错: {str(e)}"}

    def _correlation_matrix(
        self,
        data: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "相关性矩阵",
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制相关性矩阵热力图"""
        try:
            # 统一参数获取
            figsize = kwargs.get("figsize", (10, 8))
            dpi = kwargs.get("dpi", 300)
            title_fontsize = kwargs.get("title_fontsize", 16)
            label_fontsize = kwargs.get("label_fontsize", 12)
            tick_fontsize = kwargs.get("tick_fontsize", 10)
            colormap = kwargs.get("colormap", "RdBu_r")
            annotate = kwargs.get("annotate", True)
            fmt = kwargs.get("fmt", ".2f")
            filename = kwargs.get("filename")
            format = kwargs.get("format", "png")

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

            # 保存图像
            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "correlation_matrix",
                "correlation_data": correlation_matrix.to_dict(),
                "data_summary": {
                    "variables": len(labels),
                    "observations": len(data),
                    "strong_correlations": int(
                        np.sum(np.abs(correlation_matrix.values) > 0.7) - len(labels)
                    ),
                },
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"相关性矩阵已保存到文件: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result

        except Exception as e:
            return {"error": f"相关性矩阵绘制出错: {str(e)}"}

    def _multi_series_line_chart(
        self,
        x_data: List[float],
        y_data_series: List[List[float]],
        **kwargs,
    ) -> Dict[str, Any]:
        """绘制多系列线图"""
        try:
            title = kwargs.get("title", "多系列线图")
            xlabel = kwargs.get("xlabel", "X轴")
            ylabel = kwargs.get("ylabel", "Y轴")
            colors = kwargs.get("colors")
            line_styles = kwargs.get("line_styles")
            markers = kwargs.get("markers")
            series_labels = kwargs.get("series_labels")
            figsize = kwargs.get("figsize", (12, 6))
            dpi = kwargs.get("dpi", 300)
            style = kwargs.get("style", "whitegrid")
            title_fontsize = kwargs.get("title_fontsize", 16)
            label_fontsize = kwargs.get("label_fontsize", 12)
            tick_fontsize = kwargs.get("tick_fontsize", 10)
            grid = kwargs.get("grid", True)
            legend = kwargs.get("legend", True)
            filename = kwargs.get("filename")
            format = kwargs.get("format", "png")

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

            # 保存图像
            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "multi_series_line_chart",
                "data_summary": {
                    "series_count": num_series,
                    "data_points_per_series": len(x_data),
                    "series_labels": series_labels,
                },
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"多系列线图已保存到文件: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result

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
        """绘制函数图像工具"""
        try:
            # 导入sympy用于符号计算
            import sympy as sp
            import numpy as np
            import matplotlib.pyplot as plt

            # --- 改进解析：使用 sympify 支持任意变量与常量 pi/e ---
            try:
                expr_str = function_expression.replace("^", "**")

                # 先用 sympify 解析表达式，提供常量 pi/e
                expr = sp.sympify(expr_str, locals={"pi": sp.pi, "e": sp.E})

                # 自动检测/选择变量
                if variable is None:
                    variable = "x"

                free_syms = list(expr.free_symbols)

                if (variable == "x" and sp.Symbol("x") not in free_syms) and free_syms:
                    # 用户未显式提供 variable 且表达式没有 x，则取第一个自由符号
                    sym_var = free_syms[0]
                    variable = str(sym_var)
                else:
                    sym_var = sp.Symbol(variable)

                # 转换为可计算函数
                f = sp.lambdify(sym_var, expr, "numpy")

                # 如果要计算导数
                if derivative_order is not None and derivative_order > 0:
                    derivative_expr = expr
                    for _ in range(derivative_order):
                        derivative_expr = sp.diff(derivative_expr, sym_var)
                    df = sp.lambdify(sym_var, derivative_expr, "numpy")
                else:
                    df = None

            except Exception as e:
                return {"error": f"函数表达式解析错误: {str(e)}"}

            # 绘制函数图像
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # 生成x值和对应的y值
            x = np.linspace(x_range[0], x_range[1], num_points)
            try:
                y = f(x)

                # 处理无穷大和NaN值
                y = np.where(np.isfinite(y), y, np.nan)

                # 绘制主函数
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=line_width,
                    alpha=alpha,
                    linestyle=line_style,
                    marker=marker,
                    markersize=marker_size,
                    label=function_expression,
                )

                # 如果要绘制导数
                if df is not None:
                    dy = df(x)
                    # 处理无穷大和NaN值
                    dy = np.where(np.isfinite(dy), dy, np.nan)
                    ax.plot(
                        x,
                        dy,
                        color="red",
                        linewidth=line_width * 0.8,
                        alpha=alpha * 0.8,
                        linestyle="--",
                        label=f"{derivative_order}阶导数",
                    )

                # 如果要显示临界点
                if show_critical_points and df is not None:
                    # 找出导数接近0的点（可能是极值点）
                    dy = df(x)
                    # 寻找导数符号变化的点
                    critical_indices = np.where(np.diff(np.signbit(dy)))[0]

                    for idx in critical_indices:
                        critical_x = x[idx]
                        critical_y = f(critical_x)
                        ax.plot(critical_x, critical_y, "ro", markersize=6)
                        ax.annotate(
                            f"({critical_x:.2f}, {critical_y:.2f})",
                            (critical_x, critical_y),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha="center",
                        )

            except Exception as e:
                return {"error": f"函数计算错误: {str(e)}"}

            # 设置图表属性
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            # 显示网格
            if grid:
                ax.grid(alpha=grid_alpha)

            # 在图表上显示方程
            if show_equation:
                equation_text = f"$f({variable}) = {sp.latex(expr)}$"
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)

                # 设置方程位置
                if equation_position == "upper right":
                    pos_x, pos_y = 0.95, 0.95
                    ha, va = "right", "top"
                elif equation_position == "upper left":
                    pos_x, pos_y = 0.05, 0.95
                    ha, va = "left", "top"
                elif equation_position == "lower right":
                    pos_x, pos_y = 0.95, 0.05
                    ha, va = "right", "bottom"
                elif equation_position == "lower left":
                    pos_x, pos_y = 0.05, 0.05
                    ha, va = "left", "bottom"
                else:
                    pos_x, pos_y = 0.5, 0.95
                    ha, va = "center", "top"

                ax.text(
                    pos_x,
                    pos_y,
                    equation_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment=va,
                    horizontalalignment=ha,
                    bbox=props,
                )

            # 添加图例
            if df is not None:
                ax.legend()

            # 保存图像到文件
            file_path = self._save_figure(fig, format, filename)

            # 准备返回结果
            result = {
                "function": function_expression,
                "variable": variable,
                "x_range": x_range,
                "points": num_points,
            }

            # 添加文件路径信息
            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"函数图像已保存到文件: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "图表保存失败"
                result["success"] = False

            return result

        except Exception as e:
            return {"error": f"函数绘图出错: {str(e)}"}
