# -*- coding: utf-8 -*-
"""
Statistical Plotting Module
Provides comprehensive statistical chart plotting functionality, saving images directly to a specified path
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
    """Set up language-specific font support"""

    # Read environment variable FONT_PATH
    font_path = os.getenv("FONT_PATH")
    if font_path and os.path.exists(font_path):
        # Add font to the manager to ensure availability
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        # Use default system Chinese font
        plt.rcParams["font.sans-serif"] = [
            "SimHei"
        ]  # For proper display of Chinese labels
        plt.rcParams["axes.unicode_minus"] = False  # For correct display of minus sign


# Initialize font settings
setup_font()


class PlottingCalculator:
    """Statistical plotting calculator providing complete chart plotting capability"""

    def __init__(self):
        """Initialize the plotting calculator"""

        # Set default style
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
        title: str = "Statistical Chart",
        xlabel: str = "X Axis",
        ylabel: str = "Y Axis",
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
        Comprehensive statistical plotting tool - supports multiple chart types

        Args:
            chart_type: Chart type ('bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap', 'correlation_matrix', 'multi_series_line')
            data: Single data group (used for bar, pie, histogram, box plot)
            x_data: X-axis data (used for line, scatter, multi-series line)
            y_data: Y-axis data (used for line, scatter)
            y_data_series: Multi-series Y data (for multi-series lines)
            series_labels: Labels for multi-series charts
            matrix_data: Matrix data (for heatmap, grouped boxplot, correlation matrix)
            labels: Data labels
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output chart filename (optional)
            format: Chart output format (png, jpg, svg etc.)
            colors: Color list
            figsize: Figure size (width, height)
            dpi: Figure resolution
            style: Chart style
            show_values: Whether to display values (bar chart)
            horizontal: Horizontal chart (bar chart)
            trend_line: Show trend line (scatter)
            trend_line_color: Trend line color
            trend_line_equation: Trend line equation
            bins: Bins for histogram
            annotate: Show annotation (heatmap)
            colormap: Colormap (heatmap)
            grid: Show grid
            **kwargs: Other chart parameters

        Returns:
            Result dictionary containing image save information
        """
        try:
            # Set default figure size
            actual_figsize = figsize
            if actual_figsize is None:
                if chart_type == "pie":
                    actual_figsize = (8, 8)
                elif chart_type == "heatmap":
                    actual_figsize = (10, 8)
                else:
                    actual_figsize = (10, 6)

            # Basic universal parameters
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
                    return {"error": "Box plot requires data or matrix_data parameter"}

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
                return {
                    "error": f"Unsupported chart type: {chart_type} or missing required data parameter"
                }

        except Exception as e:
            return {"error": f"Statistical plotting error: {str(e)}"}

    def _prepare_figure(
        self, figsize: Tuple[float, float], dpi: int, style: str
    ) -> plt.Figure:
        """Prepare figure object"""
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
        """Save figure to file

        Args:
            fig: Matplotlib figure object
            format: Image format (png, jpg, svg etc.)
            custom_filename: Custom filename (optional, if not provided use default)

        Returns:
            str: File path (on success) or None (on failure)
        """
        try:
            # If no custom filename is given, generate a default one
            if custom_filename is None:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_filename = f"chart_{timestamp}"

            # Use utility function to generate a unique filename
            file_path, filename = generate_unique_filename(
                "chart", format, custom_filename
            )

            # Save image to file
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
                logging.error(f"Error saving image to file: {str(e)}")
                return None

        finally:
            # Always clean up resources
            try:
                plt.close(fig)
            except:
                pass
            # Force garbage collection
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
        """Apply common style settings"""
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
        title: str = "Bar Chart",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw bar chart"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            colors = kwargs.get("colors", self.default_colors[: len(data)])
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if labels is None:
                labels = [f"Category{i+1}" for i in range(len(data))]

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
                result["message"] = (
                    f"Bar chart saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Bar chart plotting error: {str(e)}"}

    def _pie_chart(
        self,
        data: List[float],
        labels: Optional[List[str]] = None,
        title: str = "Pie Chart",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw pie chart"""
        try:
            figsize = kwargs.get("figsize", (8, 8))
            dpi = kwargs.get("dpi", 300)
            colors = kwargs.get("colors", self.default_colors[: len(data)])
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if labels is None:
                labels = [f"Category{i+1}" for i in range(len(data))]

            ax.pie(data, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title(title, fontsize=16)

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "pie_chart",
                "data_summary": {"categories": len(data), "total": sum(data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Pie chart saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Pie chart plotting error: {str(e)}"}

    def _line_chart(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Line Chart",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw line chart"""
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
            ax.set_xlabel(kwargs.get("xlabel", "X Axis"))
            ax.set_ylabel(kwargs.get("ylabel", "Y Axis"))

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "line_chart",
                "data_summary": {"points": len(x_data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Line chart saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Line chart plotting error: {str(e)}"}

    def _scatter_plot(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Scatter Plot",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw scatter plot"""
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
            ax.set_xlabel(kwargs.get("xlabel", "X Axis"))
            ax.set_ylabel(kwargs.get("ylabel", "Y Axis"))

            # Add trend line if specified
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

                    # Add equation if provided
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
                    print(f"Trend line plotting failed: {str(e)}")

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "scatter_plot",
                "data_summary": {"points": len(x_data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Scatter plot saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Scatter plot plotting error: {str(e)}"}

    def _histogram(
        self, data: List[float], title: str = "Histogram", **kwargs
    ) -> Dict[str, Any]:
        """Draw histogram"""
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
            ax.set_xlabel(kwargs.get("xlabel", "Value"))
            ax.set_ylabel(kwargs.get("ylabel", "Frequency"))

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "histogram",
                "data_summary": {"bins": len(bins_edges) - 1, "total_count": len(data)},
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Histogram saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Histogram plotting error: {str(e)}"}

    def _box_plot(
        self,
        data: Union[List[float], List[List[float]]],
        title: str = "Box Plot",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw box plot"""
        try:
            figsize = kwargs.get("figsize", (10, 6))
            dpi = kwargs.get("dpi", 300)
            filename = kwargs.get("filename", None)
            format = kwargs.get("format", "png")

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if isinstance(data[0], list):
                # Multiple groups
                ax.boxplot(data)
                ax.set_title(title, fontsize=16)
                ax.set_xlabel(kwargs.get("xlabel", "Group"))
                ax.set_ylabel(kwargs.get("ylabel", "Value"))
                data_summary = {"groups": len(data)}
            else:
                # Single group
                ax.boxplot(data)
                ax.set_title(title, fontsize=16)
                ax.set_ylabel(kwargs.get("ylabel", "Value"))
                data_summary = {"values": len(data)}

            file_path = self._save_figure(fig, format, filename)

            result = {
                "chart_type": "box_plot",
                "data_summary": data_summary,
            }

            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Box plot saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Box plot plotting error: {str(e)}"}

    def _heatmap(
        self, data: List[List[float]], title: str = "Heatmap", **kwargs
    ) -> Dict[str, Any]:
        """Draw heatmap"""
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

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Add value annotations
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
                result["message"] = (
                    f"Heatmap saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result
        except Exception as e:
            return {"error": f"Heatmap plotting error: {str(e)}"}

    def _correlation_matrix(
        self,
        data: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw correlation matrix heatmap"""
        try:
            # Unified parameter retrieval
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

            # Convert to DataFrame for easier correlation computation
            if labels is None:
                labels = [f"Variable{i+1}" for i in range(len(data[0]))]

            df = pd.DataFrame(data, columns=labels)
            correlation_matrix = df.corr()

            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Draw heatmap
            im = ax.imshow(
                correlation_matrix.values, cmap=colormap, vmin=-1, vmax=1, aspect="auto"
            )

            # Set ticks and labels
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=tick_fontsize)
            ax.set_yticklabels(labels, fontsize=tick_fontsize)

            # Rotate x axis labels
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Add value annotations
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

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Correlation Coefficient", fontsize=label_fontsize)

            # Set title
            ax.set_title(title, fontsize=title_fontsize, pad=20)

            # Adjust layout
            plt.tight_layout()

            # Save figure
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
                    f"Correlation matrix saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result

        except Exception as e:
            return {"error": f"Correlation matrix plotting error: {str(e)}"}

    def _multi_series_line_chart(
        self,
        x_data: List[float],
        y_data_series: List[List[float]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Draw multi-series line chart"""
        try:
            title = kwargs.get("title", "Multi-Series Line Chart")
            xlabel = kwargs.get("xlabel", "X Axis")
            ylabel = kwargs.get("ylabel", "Y Axis")
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

            # Set style
            if style:
                sns.set_style(style)
                setup_font()

            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Set default parameters
            num_series = len(y_data_series)
            if series_labels is None:
                series_labels = [f"Series{i+1}" for i in range(num_series)]

            if colors is None:
                colors = self.default_colors[:num_series]

            if line_styles is None:
                line_styles = ["-"] * num_series

            if markers is None:
                markers = ["o", "s", "^", "D", "v"] * (num_series // 5 + 1)

            # Plot each series
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

            # Apply styling
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

            # Save figure
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
                    f"Multi-series line chart saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result

        except Exception as e:
            return {"error": f"Multi-series line chart plotting error: {str(e)}"}

    def plot_function_tool(
        self,
        function_expression: str,
        variable: str = "x",
        x_range: Tuple[float, float] = (-10, 10),
        num_points: int = 1000,
        title: str = "Function Plot",
        xlabel: str = "X Axis",
        ylabel: str = "Y Axis",
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
        """Function plotting tool"""
        try:
            # Import sympy for symbolic computation
            import sympy as sp
            import numpy as np
            import matplotlib.pyplot as plt

            # --- Improved parsing: use sympify to support arbitrary variable and constants pi/e ---
            try:
                expr_str = function_expression.replace("^", "**")

                # Parse expression using sympify, provide constants pi/e
                expr = sp.sympify(expr_str, locals={"pi": sp.pi, "e": sp.E})

                # Automatically detect/select variable
                if variable is None:
                    variable = "x"

                free_syms = list(expr.free_symbols)

                if (variable == "x" and sp.Symbol("x") not in free_syms) and free_syms:
                    # User did not explicitly specify variable and expression does not have x, use first free symbol
                    sym_var = free_syms[0]
                    variable = str(sym_var)
                else:
                    sym_var = sp.Symbol(variable)

                # Convert to numerical function
                f = sp.lambdify(sym_var, expr, "numpy")

                # If derivative is needed
                if derivative_order is not None and derivative_order > 0:
                    derivative_expr = expr
                    for _ in range(derivative_order):
                        derivative_expr = sp.diff(derivative_expr, sym_var)
                    df = sp.lambdify(sym_var, derivative_expr, "numpy")
                else:
                    df = None

            except Exception as e:
                return {"error": f"Function expression parsing error: {str(e)}"}

            # Plot function
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Generate x values and corresponding y values
            x = np.linspace(x_range[0], x_range[1], num_points)
            try:
                y = f(x)

                # Handle Inf and NaN
                y = np.where(np.isfinite(y), y, np.nan)

                # Plot main function
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

                # If plotting derivative
                if df is not None:
                    dy = df(x)
                    # Handle Inf and NaN
                    dy = np.where(np.isfinite(dy), dy, np.nan)
                    ax.plot(
                        x,
                        dy,
                        color="red",
                        linewidth=line_width * 0.8,
                        alpha=alpha * 0.8,
                        linestyle="--",
                        label=f"{derivative_order} Order Derivative",
                    )

                # If showing critical points
                if show_critical_points and df is not None:
                    # Find points where derivative is close to zero (potential extrema)
                    dy = df(x)
                    # Find sign changes in the derivative
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
                return {"error": f"Function computation error: {str(e)}"}

            # Set plot attributes
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            # Show grid
            if grid:
                ax.grid(alpha=grid_alpha)

            # Show equation on the plot
            if show_equation:
                equation_text = f"$f({variable}) = {sp.latex(expr)}$"
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)

                # Set equation position
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

            # Add legend
            if df is not None:
                ax.legend()

            # Save figure to file
            file_path = self._save_figure(fig, format, filename)

            # Prepare result
            result = {
                "function": function_expression,
                "variable": variable,
                "x_range": x_range,
                "points": num_points,
            }

            # Add file path info
            if file_path:
                result["file_path"] = file_path
                result["message"] = (
                    f"Function plot saved to file: {os.path.basename(file_path)}"
                )
                result["success"] = True
            else:
                result["error"] = "Chart saving failed"
                result["success"] = False

            return result

        except Exception as e:
            return {"error": f"Function plotting error: {str(e)}"}
