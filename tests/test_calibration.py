"""Tests for mars calibration module."""

import numpy as np
import pandas as pd
import pytest

from mars.calibration import MzCalibrator


class TestMzCalibrator:
    """Tests for MzCalibrator class."""

    @pytest.fixture
    def sample_matches(self) -> pd.DataFrame:
        """Create sample match data for testing."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame(
            {
                "precursor_mz": np.random.uniform(400, 1000, n),
                "fragment_mz": np.random.uniform(200, 1200, n),
                "absolute_time": np.random.uniform(0, 3600, n),
                "log_tic": np.log10(np.random.uniform(1e6, 1e8, n)),
                "log_intensity": np.log10(np.random.uniform(500, 10000, n)),
                "injection_time": np.random.uniform(10, 150, n) / 1000.0,
                "tic": np.random.uniform(1e6, 1e8, n),
                "observed_intensity": np.random.uniform(500, 10000, n),
                "delta_mz": np.random.normal(0.05, 0.1, n),
            }
        )
        df["tic_injection_time"] = df["tic"] * df["injection_time"]
        return df

    def test_calibrator_init(self):
        """Test MzCalibrator initialization."""
        calibrator = MzCalibrator()
        assert calibrator.model is None
        # feature_names includes optional features
        assert "precursor_mz" in calibrator.feature_names
        assert "fragment_mz" in calibrator.feature_names
        assert "log_tic" in calibrator.feature_names

    def test_calibrator_fit(self, sample_matches):
        """Test model training."""
        calibrator = MzCalibrator(n_estimators=10)  # Small for fast test
        calibrator.fit(sample_matches)

        assert calibrator.model is not None
        assert "train_mae" in calibrator.training_stats
        assert "train_rmse" in calibrator.training_stats
        assert "feature_importance" in calibrator.training_stats

    def test_calibrator_predict(self, sample_matches):
        """Test prediction using DataFrame interface."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        # Predict on same data using DataFrame
        corrections = calibrator.predict(matches=sample_matches)

        assert len(corrections) == len(sample_matches)
        assert isinstance(corrections, np.ndarray)

    def test_calibrator_predict_before_fit(self):
        """Test that predict fails before fit."""
        calibrator = MzCalibrator()
        df = pd.DataFrame({"fragment_mz": [500.0]})

        with pytest.raises(ValueError, match="Model not trained"):
            calibrator.predict(matches=df)

    def test_calibrator_save_load(self, sample_matches, tmp_path):
        """Test model save/load."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        # Save
        model_path = tmp_path / "model.pkl"
        calibrator.save(model_path)
        assert model_path.exists()

        # Load into new calibrator
        loaded = MzCalibrator.load(model_path)
        assert loaded.model is not None
        assert loaded.feature_names == calibrator.feature_names

        # Predictions should match
        test_input = sample_matches.iloc[:2].copy()

        orig_pred = calibrator.predict(matches=test_input)
        loaded_pred = loaded.predict(matches=test_input)
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_create_calibration_function(self, sample_matches):
        """Test calibration function creation."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        cal_func = calibrator.create_calibration_function()

        # Test with sample data
        metadata = {
            "precursor_mz": 500.0,
            "tic": 1e7,
            "absolute_time": 100.0,
            "injection_time": 0.05,
        }
        mz_array = np.array([500.0, 600.0, 700.0])
        intensity_array = np.array([1000.0, 2000.0, 3000.0])

        calibrated = cal_func(metadata, mz_array, intensity_array)

        assert len(calibrated) == len(mz_array)
        assert isinstance(calibrated, np.ndarray)

    def test_get_stats_summary(self, sample_matches):
        """Test stats summary generation."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        summary = calibrator.get_stats_summary()

        assert isinstance(summary, str)
        assert "Calibration Model Summary" in summary
        assert "Train MAE" in summary
        assert "Feature importance" in summary

    def test_missing_injection_time_removal(self):
        """Test that injection_time is removed when universally missing."""
        np.random.seed(42)
        n = 100

        # Create matches WITHOUT injection_time
        df = pd.DataFrame(
            {
                "precursor_mz": np.random.uniform(400, 1000, n),
                "fragment_mz": np.random.uniform(200, 1200, n),
                "absolute_time": np.random.uniform(0, 3600, n),
                "log_tic": np.log10(np.random.uniform(1e6, 1e8, n)),
                "log_intensity": np.log10(np.random.uniform(500, 10000, n)),
                "injection_time": None,  # Universally missing
                "tic_injection_time": None,  # Will be removed too
                "delta_mz": np.random.normal(0.05, 0.1, n),
            }
        )

        calibrator = MzCalibrator(n_estimators=5)
        calibrator.fit(df)

        # Check that injection_time was NOT included
        assert "injection_time" not in calibrator.feature_names
        assert "tic_injection_time" not in calibrator.feature_names

    def test_sparse_injection_time_rows_dropped(self):
        """Test that rows with missing injection_time are dropped when sparse."""
        np.random.seed(42)
        n = 100

        # Create matches with SOME injection_time missing
        df = pd.DataFrame(
            {
                "precursor_mz": np.random.uniform(400, 1000, n),
                "fragment_mz": np.random.uniform(200, 1200, n),
                "absolute_time": np.random.uniform(0, 3600, n),
                "log_tic": np.log10(np.random.uniform(1e6, 1e8, n)),
                "log_intensity": np.log10(np.random.uniform(500, 10000, n)),
                "injection_time": np.where(
                    np.random.random(n) < 0.8, np.random.uniform(10, 150, n) / 1000.0, None
                ),
                "tic": np.random.uniform(1e6, 1e8, n),
                "tic_injection_time": None,  # Will be computed or None
                "delta_mz": np.random.normal(0.05, 0.1, n),
            }
        )
        # Add tic_injection_time where injection_time exists
        df["tic_injection_time"] = df.apply(
            lambda row: row["tic"] * row["injection_time"]
            if row["injection_time"] is not None
            else None,
            axis=1,
        )

        n_before = len(df)
        calibrator = MzCalibrator(n_estimators=5)
        calibrator.fit(df)

        # Check that injection_time IS included
        assert "injection_time" in calibrator.feature_names
