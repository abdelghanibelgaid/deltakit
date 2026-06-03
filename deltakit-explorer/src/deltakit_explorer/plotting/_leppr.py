# (c) Copyright Riverlane 2020-2025.
"""Plotting helpers for logical-error-probability-per-round results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting._utils import get_figure_and_axes
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)
from deltakit_explorer.plotting.results import interpolate_leppr


def _plot_leppr_result(
    leppr_result: LEPPRResult,
    *,
    ax: Axes,
    title: str | None = None,
) -> None:
    """Plot an interpolated LEPPR result on existing axes."""
    ax.plot(
        leppr_result.rounds,
        leppr_result.interpolated,
        label=leppr_result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        leppr_result.rounds,
        leppr_result.lower_boundary,
        leppr_result.upper_boundary,
        label=leppr_result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title(title if title is not None else "Logical Error Probability per Round")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()


def _plot_logical_error_probability_observations(
    *,
    ax: Axes,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: npt.NDArray[np.float64] | Sequence[float] | None,
    num_sigmas: int,
) -> None:
    """Plot observed logical error probabilities on existing axes."""
    lens = {len(num_rounds), len(logical_error_probability)}
    if logical_error_probability_stddev is not None:
        lens.add(len(logical_error_probability_stddev))
    if len(lens) > 1:
        msg = (
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. Got the following "
            f"lengths: {lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(num_rounds)
    sorted_num_rounds = np.asarray(num_rounds)[isort]
    sorted_logical_error_probability = np.asarray(logical_error_probability)[isort]
    sorted_logical_error_probability_stddev = None
    if logical_error_probability_stddev is not None:
        sorted_logical_error_probability_stddev = (
            num_sigmas * np.asarray(logical_error_probability_stddev)[isort]
        )

    ax.errorbar(
        sorted_num_rounds,
        sorted_logical_error_probability,
        yerr=sorted_logical_error_probability_stddev,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
    )


def plot_logical_error_probability_per_round(
    leppr_data: LEPPRData | LEPPRResult,
    num_rounds: npt.NDArray[np.int_] | Sequence[int] | None = None,
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float] | None = None,
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot logical error probability per round data or an interpolated result.

    This specialised function owns the LEPPR rendering logic. When passed
    ``LEPPRData``, it plots the observed logical error probabilities, computes
    the interpolated LEPPR fit, and plots the fitted curve with confidence
    bounds. When passed ``LEPPRResult``, it plots only the already-interpolated
    fit data.

    Args:
        leppr_data: LEPPR fit data or an already-interpolated LEPPR result.
        num_rounds: A sequence of integers representing the number of rounds
            used to get the corresponding logical error probabilities. Required
            when ``leppr_data`` is ``LEPPRData``.
        logical_error_probability: A sequence of logical error probabilities
            corresponding to ``num_rounds``. Required when ``leppr_data`` is
            ``LEPPRData``.
        logical_error_probability_stddev: A sequence of standard deviations for
            the logical error probabilities. If None, no error bars will be
            plotted. Default is None.
        num_sigmas: Number of sigmas to consider when plotting error bars.
        num_points: Number of interpolation points. Default 200.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.
        title: An optional custom title for the plot. If None, the default
            LEPPR title will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_stddev,
        ...     compute_logical_error_per_round,
        ... )
        >>> num_failed_shots = [34, 151, 356]
        >>> num_shots = [500000] * 3
        >>> num_rounds = [2, 4, 6]
        >>> res = compute_logical_error_per_round(
        ...     num_failed_shots=num_failed_shots,
        ...     num_shots=num_shots,
        ...     num_rounds=num_rounds,
        ... )
        >>> lep, lep_stddev = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_stddev=lep_stddev,
        ... )
    """
    fig, ax = get_figure_and_axes(fig, ax)

    if isinstance(leppr_data, LEPPRResult):
        leppr_result = leppr_data
        if num_rounds is not None or logical_error_probability is not None:
            if num_rounds is None or logical_error_probability is None:
                msg = (
                    "'num_rounds' and 'logical_error_probability' must either both "
                    "be provided or both be omitted."
                )
                raise ValueError(msg)
            _plot_logical_error_probability_observations(
                ax=ax,
                num_rounds=num_rounds,
                logical_error_probability=logical_error_probability,
                logical_error_probability_stddev=logical_error_probability_stddev,
                num_sigmas=num_sigmas,
            )
    else:
        if num_rounds is None or logical_error_probability is None:
            msg = (
                "'num_rounds' and 'logical_error_probability' are required when "
                "plotting 'LogicalErrorProbabilityPerRoundData'."
            )
            raise ValueError(msg)

        _plot_logical_error_probability_observations(
            ax=ax,
            num_rounds=num_rounds,
            logical_error_probability=logical_error_probability,
            logical_error_probability_stddev=logical_error_probability_stddev,
            num_sigmas=num_sigmas,
        )
        leppr_result = interpolate_leppr(
            leppr_data, num_sigmas=num_sigmas, num_points=num_points
        )

    _plot_leppr_result(leppr_result, ax=ax, title=title)
    return fig, ax
