# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_logical_error_probability_per_round
from deltakit_explorer.plotting.results import LambdaResult
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)


def plot(
    result: LambdaResult | LEPPRResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Dispatch to specialised plotting based on the result type.

    Args:
        result: The precomputed plot data.
        fig: An existing matplotlib Figure. If None, a new figure will be
            created by the specialised plotting function. Default is None.
        ax: An existing matplotlib Axes. If None, a new axes will be created by
            the specialised plotting function. Default is None.
        title: An optional custom title for the plot. If None, a default title
            based on the result type will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        TypeError: If the ``result`` type is not supported.

    Examples:

        Plotting a Lambda fit curve::

            from deltakit_explorer.plotting.results import interpolate_lambda

            lambda_result = interpolate_lambda(lambda_data)
            fig, ax = plot(lambda_result)

        Plotting a LEPPR fit curve::

            from deltakit_explorer.plotting.results import interpolate_leppr

            leppr_result = interpolate_leppr(leppr_data)
            fig, ax = plot(leppr_result)

    """
    match result:
        case LambdaResult():
            return plot_lambda(result, fig=fig, ax=ax, title=title)
        case LEPPRResult():
            return plot_logical_error_probability_per_round(
                result, fig=fig, ax=ax, title=title
            )
        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected `LambdaResult` or `LEPPRResult`."
            )
            raise TypeError(msg)
