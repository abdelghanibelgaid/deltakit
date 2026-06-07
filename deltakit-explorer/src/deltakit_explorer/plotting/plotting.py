# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_leppr
from deltakit_explorer.plotting.results import (
    LambdaResult,
    interpolate_lambda,
    interpolate_leppr,
)
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)


def plot(
    result: LambdaData | LambdaResult | LEPPRData | LEPPRResult,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Prepare plot data and dispatch to the specialised plotter.

    This high-level plotting function accepts either raw analysis data or an
    already-interpolated plotting result. Raw analysis data is interpolated here
    before being dispatched to the specialised renderer for its result type.

    Args:
        result: Raw analysis data or precomputed plot data.
        num_sigmas: Number of standard deviations for the error band when
            interpolation is required. Default 3.
        num_points: Number of interpolation points when interpolation is
            required. Default 200.
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

        Plotting raw Lambda data::

            fig, ax = plot(lambda_data)

        Plotting a precomputed Lambda result::

            from deltakit_explorer.plotting.results import interpolate_lambda

            lambda_result = interpolate_lambda(lambda_data)
            fig, ax = plot(lambda_result)

        Plotting raw logical error probability per round data::

            fig, ax = plot(leppr_data)

        Plotting a precomputed LEPPR fit curve::

            from deltakit_explorer.plotting.results import interpolate_leppr

            leppr_result = interpolate_leppr(leppr_data)
            fig, ax = plot(leppr_result)

    """
    match result:
        case LambdaData():
            lambda_result = interpolate_lambda(
                result, num_sigmas=num_sigmas, num_points=num_points
            )
            return plot_lambda(lambda_result, fig=fig, ax=ax, title=title)
        case LambdaResult():
            return plot_lambda(result, fig=fig, ax=ax, title=title)
        case LEPPRData():
            leppr_result = interpolate_leppr(
                result, num_sigmas=num_sigmas, num_points=num_points
            )
            return plot_leppr(leppr_result, fig=fig, ax=ax, title=title)
        case LEPPRResult():
            return plot_leppr(result, fig=fig, ax=ax, title=title)
        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected `LambdaData`, `LambdaResult`, "
                "`LogicalErrorProbabilityPerRoundData`, or `LEPPRResult`."
            )
            raise TypeError(msg)
