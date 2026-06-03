# (c) Copyright Riverlane 2020-2025.
"""Plotting helpers for error-suppression factor results."""

from __future__ import annotations

from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.plotting._utils import get_figure_and_axes
from deltakit_explorer.plotting.results import LambdaResult, interpolate_lambda


def _plot_lambda_result(
    lambda_result: LambdaResult,
    *,
    ax: Axes,
    title: str | None = None,
) -> None:
    """Plot an interpolated Lambda result on existing axes."""
    ax.plot(
        lambda_result.distances,
        lambda_result.interpolated,
        label=lambda_result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        lambda_result.distances,
        lambda_result.lower_boundary,
        lambda_result.upper_boundary,
        label=lambda_result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title(title if title is not None else "Error Suppression Factor Λ")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Logical Error Probability per Round")
    ax.legend()


def plot_lambda(
    lambda_data: LambdaData | LambdaResult,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot Λ-fitted data or an interpolated Lambda plotting result.

    This specialised function owns the Lambda rendering logic. When passed
    ``LambdaData``, it plots the observed logical error-probability-per-round
    points with error bars, interpolates the Lambda fit, and plots the fitted
    curve with confidence bounds. When passed ``LambdaResult``, it plots only
    the already-interpolated fit data.

    Args:
        lambda_data: Lambda fit data or an already-interpolated Lambda result.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.
        title: An optional custom title for the plot. If None, the default
            Lambda title will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:
        from deltakit_explorer.analysis import calculate_lambda_and_lambda_std

        lambda_data = calculate_lambda_and_lambda_std(
            distances=[5, 7, 9],
            leppr=[0.15, 0.1, 0.05],
            leppr_stddev=[0.01, 0.008, 0.005],
        )
        fig, ax = plot_lambda(
            lambda_data=lambda_data,
        )
        ax.set_yscale("log")
        plt.show()
    """
    fig, ax = get_figure_and_axes(fig, ax)

    if isinstance(lambda_data, LambdaResult):
        lambda_result = lambda_data
    else:
        ax.errorbar(
            lambda_data.distances,
            lambda_data.leppr,
            yerr=lambda_data.leppr_std * num_sigmas,
            fmt=".",
            color=RIVERLANE_PLOT_COLOURS[1],
            label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
        )
        lambda_result = interpolate_lambda(
            lambda_data, num_sigmas=num_sigmas, num_points=num_points
        )

    _plot_lambda_result(lambda_result, ax=ax, title=title)
    return fig, ax
