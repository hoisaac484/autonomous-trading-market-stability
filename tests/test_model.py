import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from herding_abm.data import clean_adjusted_closes, rebuild_from_raw
from herding_abm.model import Scenario, calibration, simulate
from herding_abm.reproducibility import verify_primary_results


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"steps": 90, "shock_start": 35, "shock_duration": 10, "n_agents": 60}


class ModelTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_finite(self):
        calib = calibration(None)
        scenario = Scenario(seed=17)
        first, first_metrics = simulate(CONFIG, scenario, calib)
        second, second_metrics = simulate(CONFIG, scenario, calib)
        self.assertTrue(first.equals(second))
        self.assertEqual(first_metrics, second_metrics)
        self.assertTrue(np.isfinite(first.select_dtypes("number").to_numpy()).all())
        prices = first[[column for column in first if column.startswith("price_")]]
        self.assertTrue((prices > 0).all().all())

    def test_both_market_mechanisms_return_required_metrics(self):
        required = {"realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
                    "peak_herding", "mean_depth", "recovery_time"}
        for market in ("traditional", "amm"):
            frame, metrics = simulate(CONFIG, Scenario(market=market, seed=3), calibration(None))
            self.assertEqual(len(frame), CONFIG["steps"])
            self.assertTrue(required <= metrics.keys())

    def test_zero_autonomous_participation_has_zero_herding(self):
        frame, _ = simulate(CONFIG, Scenario(participation=0.0), calibration(None))
        self.assertTrue(frame["herding"].eq(0).all())

    def test_cleaning_aligns_common_dates_and_removes_invalid_values(self):
        dates = pd.date_range("2010-01-04", periods=600, freq="B")
        data = {ticker: pd.Series(np.arange(600) + 100.0, index=dates)
                for ticker in ("SPY", "QQQ", "GLD", "TLT", "USO")}
        data["USO"].iloc[20] = np.nan
        cleaned, report = clean_adjusted_closes(data, "2010-01-04", "2012-12-31")
        self.assertEqual(list(cleaned.columns), ["Date", "SPY", "QQQ", "GLD", "TLT", "USO"])
        self.assertEqual(len(cleaned), 599)
        self.assertEqual(report["rows_removed_by_common_date_alignment"], 1)

    def test_default_configuration_is_the_4320_run_design(self):
        config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
        count = np.prod([len(config[name]) for name in (
            "autonomous_participation", "similarity", "markets", "shocks", "safeguards", "seeds"
        )])
        self.assertEqual(int(count), 4320)
        self.assertEqual(config["n_agents"], 300)
        self.assertEqual(config["steps"], 500)
        self.assertEqual(config["intervals_per_day"], 24)
        self.assertIsNotNone(config["data"]["prices_csv"])

    def test_frozen_primary_results_match_dissertation_headlines(self):
        report = verify_primary_results(
            ROOT / "outputs/historical_full_methodology/experiment_runs.csv"
        )
        self.assertTrue(report["ok"], report["errors"])

    def test_seeded_canonical_path_matches_archived_result(self):
        config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
        calib = calibration(ROOT / config["data"]["prices_csv"])
        scenario = Scenario("traditional", 0.0, 0.0, "fundamental", "none", 11)
        _, metrics = simulate(config, scenario, calib)
        runs = pd.read_csv(ROOT / "outputs/historical_full_methodology/experiment_runs.csv")
        archived = runs[
            runs.market.eq("traditional") & runs.participation.eq(0) & runs.similarity.eq(0)
            & runs.shock.eq("fundamental") & runs.safeguard.eq("none") & runs.seed.eq(11)
        ].iloc[0]
        for key in ("realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
                    "normalised_liquidity_deterioration"):
            self.assertAlmostEqual(metrics[key], archived[key], places=14)

    def test_frozen_raw_data_rebuilds_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = rebuild_from_raw(ROOT / "data/raw", directory, "2010-01-04", "2025-12-31",
                                         make_plots=False)
            rebuilt = pd.read_csv(artifacts["prices"])
            frozen = pd.read_csv(ROOT / "data/etf_adjusted_close_2010_2025.csv")
            pd.testing.assert_frame_equal(rebuilt, frozen)


if __name__ == "__main__":
    unittest.main()
