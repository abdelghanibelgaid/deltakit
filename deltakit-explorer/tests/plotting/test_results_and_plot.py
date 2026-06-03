# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import (
    plot_logical_error_probability_per_round,
)
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import (
    LambdaResult,
    interpolate_lambda,
    interpolate_leppr,
)
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)

# Use non-interactive backend for CI
mpl.use("Agg")


class TestComputeLambdaPlot:
    def test_output_type(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert isinstance(result, LambdaResult)

    def test_default_num_points(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert len(result.distances) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, lambda_results):
        result = interpolate_lambda(lambda_results, num_points=50)
        assert len(result.distances) == 50

    def test_distance_range(self, lambda_results, distances):
        result = interpolate_lambda(lambda_results)
        assert result.distances[0] == pytest.approx(distances[0])
        assert result.distances[-1] == pytest.approx(distances[-1])

    def test_interpolated_values_positive(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        assert np.all(result.interpolated >= 0)

    def test_frozen_dataclass(self, lambda_results):
        result = interpolate_lambda(lambda_results)
        with pytest.raises(AttributeError):
            result.distances = np.array([1, 2, 3])


class TestComputeLepprPlot:
    def test_output_type(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert isinstance(result, LEPPRResult)

    def test_default_num_points(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert len(result.rounds) == 200
        assert len(result.interpolated) == 200
        assert len(result.lower_boundary) == 200
        assert len(result.upper_boundary) == 200

    def test_custom_num_points(self, leppr_results):
        result = interpolate_leppr(leppr_results, num_points=100)
        assert len(result.rounds) == 100

    def test_rounds_range(self, leppr_results, num_rounds):
        result = interpolate_leppr(leppr_results)
        assert result.rounds[0] == pytest.approx(num_rounds[0])
        assert result.rounds[-1] == pytest.approx(num_rounds[-1])

    def test_boundaries_clipped(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        assert np.all(result.lower_boundary >= 0)
        assert np.all(result.lower_boundary <= 1)
        assert np.all(result.upper_boundary >= 0)
        assert np.all(result.upper_boundary <= 1)

    def test_frozen_dataclass(self, leppr_results):
        result = interpolate_leppr(leppr_results)
        with pytest.raises(AttributeError):
            result.rounds = np.array([1, 2, 3])


class TestPlot:
    def test_plot_with_lambda_result(self, lambda_results):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plot(lambda_result)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Error Suppression Factor Λ"
        assert ax.get_xlabel() == "Code distance"
        plt.close(fig)

    def test_plot_dispatches_lambda_result_to_plot_lambda(
        self, lambda_results, monkeypatch
    ):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plt.subplots()

        def fake_plot_lambda(result, *, fig=None, ax=None, title=None):
            assert result is lambda_result
            assert title == "Custom title"
            return fig, ax

        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting.plot_lambda", fake_plot_lambda
        )

        returned_fig, returned_ax = plot(
            lambda_result, fig=fig, ax=ax, title="Custom title"
        )
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_with_leppr_result(self, leppr_results):
        leppr_result = interpolate_leppr(leppr_results)
        fig, ax = plot(leppr_result)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Logical Error Probability per Round"
        assert ax.get_xlabel() == "Rounds"
        plt.close(fig)

    def test_plot_dispatches_leppr_result_to_plot_leppr(
        self, leppr_results, monkeypatch
    ):
        leppr_result = interpolate_leppr(leppr_results)
        fig, ax = plt.subplots()

        def fake_plot_logical_error_probability_per_round(
            result, *, fig=None, ax=None, title=None
        ):
            assert result is leppr_result
            assert title == "Custom title"
            return fig, ax

        monkeypatch.setattr(
            "deltakit_explorer.plotting.plotting."
            "plot_logical_error_probability_per_round",
            fake_plot_logical_error_probability_per_round,
        )

        returned_fig, returned_ax = plot(
            leppr_result, fig=fig, ax=ax, title="Custom title"
        )
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_with_existing_fig_ax(self, lambda_results):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot(lambda_result, fig=fig, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_raises_on_mismatched_fig_ax(self, lambda_results):
        lambda_result = interpolate_lambda(lambda_results)
        fig, _ = plt.subplots()
        with pytest.raises(ValueError, match="both `None` or both set"):
            plot(lambda_result, fig=fig, ax=None)
        plt.close(fig)

    def test_plot_raises_on_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported result type"):
            plot("invalid")


class TestSpecialisedPlotters:
    def test_plot_lambda_accepts_interpolated_result(self, lambda_results):
        lambda_result = interpolate_lambda(lambda_results)
        fig, ax = plot_lambda(lambda_result)
        assert fig is not None
        assert ax.get_title() == "Error Suppression Factor Λ"
        assert ax.get_xlabel() == "Code distance"
        plt.close(fig)

    def test_plot_lambda_accepts_raw_lambda_data(self, lambda_results):
        fig, ax = plot_lambda(lambda_results, num_points=50)
        assert fig is not None
        assert ax.get_title() == "Error Suppression Factor Λ"
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plot_leppr_accepts_interpolated_result(self, leppr_results):
        leppr_result = interpolate_leppr(leppr_results)
        fig, ax = plot_logical_error_probability_per_round(leppr_result)
        assert fig is not None
        assert ax.get_title() == "Logical Error Probability per Round"
        assert ax.get_xlabel() == "Rounds"
        plt.close(fig)

    def test_plot_leppr_accepts_raw_leppr_data(self, leppr_results, num_rounds):
        logical_error_probability = np.array([0.01, 0.02, 0.03])
        logical_error_probability_stddev = np.array([0.001, 0.002, 0.003])
        fig, ax = plot_logical_error_probability_per_round(
            leppr_results,
            num_rounds=num_rounds,
            logical_error_probability=logical_error_probability,
            logical_error_probability_stddev=logical_error_probability_stddev,
            num_points=50,
        )
        assert fig is not None
        assert ax.get_title() == "Logical Error Probability per Round"
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plot_leppr_requires_observations_for_raw_data(self, leppr_results):
        with pytest.raises(ValueError, match="are required"):
            plot_logical_error_probability_per_round(leppr_results)
