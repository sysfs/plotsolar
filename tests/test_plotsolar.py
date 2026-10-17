import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

import plotsolar as main


class SeasonalGrowthTests(unittest.TestCase):
    def setUp(self) -> None:
        matplotlib.use("Agg")
        self.data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "value": [0.0, 310.0, 600.0],
            }
        )

    def tearDown(self) -> None:
        main.plt.close("all")

    def test_import_has_no_execution_side_effects(self) -> None:
        with (
            patch("argparse.ArgumentParser.parse_args") as parse_args,
            patch("pandas.read_csv") as read_csv,
            patch.object(main.plt, "show") as show,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            importlib.reload(main)
        parse_args.assert_not_called()
        read_csv.assert_not_called()
        show.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_pipeline_returns_reusable_results_without_mutating_input(self) -> None:
        original = self.data.copy()
        validated = main.clean_data(self.data)
        rates = main.calculate_delta(validated)
        daily = main.build_daily_series(rates)
        season, smoothed = main.smooth_data(daily)
        pd.testing.assert_frame_equal(self.data, original)
        pd.testing.assert_frame_equal(validated, original)
        self.assertNotIn("doy", daily.columns)
        self.assertEqual(len(daily), 61)
        self.assertEqual(daily.iloc[-1]["date"], pd.Timestamp("2024-03-01"))
        self.assertTrue((daily["rate"] == 10.0).all())
        self.assertEqual(smoothed.shape, (61, 2))
        np.testing.assert_allclose(smoothed[:, 1], 10.0)

        with patch.object(main.plt, "show") as show:
            figure = main.plot_season(season, smoothed)
        show.assert_not_called()
        self.assertEqual(len(figure.axes[0].lines), 2)
        self.assertEqual(figure.axes[0].get_title(), "Seasonal Growth Rate")

        prediction = main.predict_next_target(rates, smoothed)
        self.assertEqual(prediction.target_value, 1000)
        self.assertEqual(prediction.date, pd.Timestamp("2024-04-10"))
        self.assertAlmostEqual(prediction.value, 1000.0)
        self.assertTrue(prediction.reached_target)

    def test_validation_drops_nulls_sorts_and_warns(self) -> None:
        data = self.data.iloc[::-1].copy()
        data.loc[3] = [pd.NaT, np.nan]
        data.loc[0, "value"] = 1000.0
        with contextlib.redirect_stdout(io.StringIO()) as output:
            validated = main.clean_data(data)
        self.assertTrue(validated["date"].is_monotonic_increasing)
        self.assertEqual(len(validated), 3)
        self.assertIn("Values are not strictly increasing", output.getvalue())
        self.assertEqual(len(data), 4)

    def test_prediction_reports_exhausted_horizon(self) -> None:
        prediction = main.predict_next_target(
            self.data, np.array([[1.0, 0.0], [366.0, 0.0]])
        )
        self.assertFalse(prediction.reached_target)
        self.assertEqual(
            prediction.date, pd.Timestamp("2024-03-01") + pd.Timedelta(days=365 * 5)
        )
        self.assertEqual(prediction.value, 600.0)

    def test_two_observations_include_both_interval_endpoints(self) -> None:
        rates = main.calculate_delta(self.data.iloc[:2])
        daily = main.build_daily_series(rates)
        self.assertEqual(len(daily), 32)
        self.assertEqual(daily.iloc[0]["date"], pd.Timestamp("2024-01-01"))
        self.assertEqual(daily.iloc[-1]["date"], pd.Timestamp("2024-02-01"))
        self.assertTrue((daily["rate"] == 10.0).all())

    def test_cli_accepts_explicit_arguments_and_shows_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.tsv"
            self.data.to_csv(path, sep="\t", header=False, index=False)
            with (
                patch.object(main.plt, "show") as show,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                status = main.main(["--input", str(path), "--date-format", "%Y-%m-%d"])
        self.assertEqual(status, 0)
        show.assert_called_once_with()
        self.assertIn("Current value: 600.00", output.getvalue())
        self.assertIn("Target value: 1000", output.getvalue())
        self.assertIn("Predicted date: 2024-04-10", output.getvalue())

    def test_cli_no_plot_skips_plotting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.tsv"
            self.data.to_csv(path, sep="\t", header=False, index=False)
            with (
                patch.object(main, "plot_season") as plot_season,
                patch.object(main.plt, "show") as show,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                status = main.main(["--input", str(path), "--no-plot"])
        self.assertEqual(status, 0)
        plot_season.assert_not_called()
        show.assert_not_called()
        self.assertIn("Predicted date: 2024-04-10", output.getvalue())

    def test_cli_no_predict_skips_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.tsv"
            self.data.to_csv(path, sep="\t", header=False, index=False)
            for plot_enabled in (True, False):
                with (
                    self.subTest(plot_enabled=plot_enabled),
                    patch.object(main, "predict_next_target") as predict,
                    patch.object(main, "plot_season") as plot,
                    patch.object(main.plt, "show") as show,
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    plot_flag = "--plot" if plot_enabled else "--no-plot"
                    status = main.main(
                        ["--input", str(path), "--no-predict", plot_flag]
                    )
                    self.assertEqual(status, 0)
                    predict.assert_not_called()
                    self.assertEqual(plot.call_count, int(plot_enabled))
                    self.assertEqual(show.call_count, int(plot_enabled))
                    self.assertNotIn("Current value:", output.getvalue())
                    self.assertNotIn("Target value:", output.getvalue())
                    self.assertNotIn("Predicted", output.getvalue())
                    self.assertNotIn("Prediction exceeded", output.getvalue())

    def test_cli_preserves_error_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.tsv"
            cases = [
                ("invalid\t10\n", 1, "Failed to read file"),
                ("2024-01-01\t10\n", 2, "Not enough data points"),
                (
                    "2024-01-01\t10\n2024-01-01\t20\n",
                    3,
                    "No daily data generated",
                ),
            ]
            for contents, expected_status, message in cases:
                with self.subTest(status=expected_status):
                    path.write_text(contents, encoding="utf-8")
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        status = main.main(["--input", str(path)])
                    self.assertEqual(status, expected_status)
                    self.assertIn(f"[ERROR] {message}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
