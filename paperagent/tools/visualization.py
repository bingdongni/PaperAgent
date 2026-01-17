"""
Visualization and Figure Generation Module

Provides capabilities for creating publication-quality visualizations:
- Statistical plots (scatter, bar, line, histogram, box plots)
- Heatmaps and correlation matrices
- 3D plots
- Network graphs
- Custom academic figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger


class AcademicVisualizer:
    """Create publication-quality academic visualizations"""

    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Set academic defaults
        sns.set_context("paper")
        sns.set_palette("colorblind")

        self.figure_counter = 0
        self.output_dir = Path("./figures")
        self.output_dir.mkdir(exist_ok=True)

    def create_scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        groups: Optional[np.ndarray] = None,
        add_regression: bool = False,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create scatter plot

        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            groups: Optional grouping variable
            add_regression: Add regression line
            save_path: Optional save path

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        if groups is not None:
            # Color by groups
            unique_groups = np.unique(groups)
            for group in unique_groups:
                mask = groups == group
                ax.scatter(x[mask], y[mask], label=str(group), alpha=0.6, s=50)
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.6, s=50)

        if add_regression:
            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
            ax.legend()

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"scatter_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_bar_chart(
        self,
        categories: List[str],
        values: List[float],
        title: str = "Bar Chart",
        xlabel: str = "Category",
        ylabel: str = "Value",
        horizontal: bool = False,
        error_bars: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        x_pos = np.arange(len(categories))

        if horizontal:
            ax.barh(x_pos, values, xerr=error_bars, alpha=0.8, capsize=5)
            ax.set_yticks(x_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel(xlabel, fontsize=12)
        else:
            ax.bar(x_pos, values, yerr=error_bars, alpha=0.8, capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"bar_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_line_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        multiple_lines: Optional[Dict[str, np.ndarray]] = None,
        markers: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """Create line plot"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        if multiple_lines:
            for label, y_data in multiple_lines.items():
                marker = 'o' if markers else None
                ax.plot(x, y_data, marker=marker, label=label, linewidth=2, markersize=5)
            ax.legend(fontsize=10)
        else:
            marker = 'o' if markers else None
            ax.plot(x, y, marker=marker, linewidth=2, markersize=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"line_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        cmap: str = "YlOrRd",
        annotate: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """Create heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        sns.heatmap(
            data,
            annot=annotate,
            fmt='.2f' if annotate else '',
            cmap=cmap,
            xticklabels=col_labels if col_labels else 'auto',
            yticklabels=row_labels if row_labels else 'auto',
            cbar_kws={'label': 'Value'},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"heatmap_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_box_plot(
        self,
        data: List[np.ndarray],
        labels: List[str],
        title: str = "Box Plot",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> str:
        """Create box plot"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Customize colors
        colors = sns.color_palette("Set2", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"boxplot_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_histogram(
        self,
        data: np.ndarray,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 30,
        kde: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """Create histogram"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        if kde:
            sns.histplot(data, bins=bins, kde=True, ax=ax, alpha=0.7)
        else:
            ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"histogram_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None
    ) -> str:
        """Create correlation matrix heatmap"""
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"correlation_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def create_violin_plot(
        self,
        data: List[np.ndarray],
        labels: List[str],
        title: str = "Violin Plot",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> str:
        """Create violin plot"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Prepare data for seaborn
        plot_data = []
        plot_labels = []
        for label, values in zip(labels, data):
            plot_data.extend(values)
            plot_labels.extend([label] * len(values))

        df = pd.DataFrame({'Value': plot_data, 'Category': plot_labels})

        sns.violinplot(data=df, x='Category', y='Value', ax=ax, palette="Set2")

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"violin_{self.figure_counter}.png"
            self.figure_counter += 1

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)


class InteractiveVisualizer:
    """Create interactive visualizations using Plotly"""

    def __init__(self):
        self.output_dir = Path("./figures")
        self.output_dir.mkdir(exist_ok=True)

    def create_interactive_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Interactive Scatter Plot",
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create interactive scatter plot"""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            hover_data=df.columns
        )

        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )

        if save_path is None:
            save_path = self.output_dir / "interactive_scatter.html"

        fig.write_html(str(save_path))
        return str(save_path)

    def create_3d_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        title: str = "3D Scatter Plot",
        color_col: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create 3D scatter plot"""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            template="plotly_white"
        )

        if save_path is None:
            save_path = self.output_dir / "scatter_3d.html"

        fig.write_html(str(save_path))
        return str(save_path)

    def create_interactive_line(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str = "Interactive Line Plot",
        save_path: Optional[str] = None
    ) -> str:
        """Create interactive line plot"""
        fig = go.Figure()

        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            template="plotly_white",
            hovermode='x unified'
        )

        if save_path is None:
            save_path = self.output_dir / "interactive_line.html"

        fig.write_html(str(save_path))
        return str(save_path)


__all__ = ['AcademicVisualizer', 'InteractiveVisualizer']
