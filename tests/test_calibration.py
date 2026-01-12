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

        return pd.DataFrame(
            {
                "fragment_mz": np.random.uniform(200, 1200, n),
                "rt": np.random.uniform(5, 30, n),
                "tic": np.random.uniform(1e6, 1e8, n),
                "observed_intensity": np.random.uniform(500, 10000, n),
                "delta_mz": np.random.normal(0.05, 0.1, n),  # Simulated bias
            }
        )

    def test_calibrator_init(self):
        """Test MzCalibrator initialization."""
        calibrator = MzCalibrator()
        assert calibrator.model is None
        assert calibrator.feature_names == [
            "fragment_mz",
            "rt",
            "log_tic",
            "log_intensity",
        ]

    def test_calibrator_fit(self, sample_matches):
        """Test model training."""
        calibrator = MzCalibrator(n_estimators=10)  # Small for fast test
        calibrator.fit(sample_matches)

        assert calibrator.model is not None
        assert "train_mae" in calibrator.training_stats
        assert "train_rmse" in calibrator.training_stats
        assert "feature_importance" in calibrator.training_stats

    def test_calibrator_predict(self, sample_matches):
        """Test prediction."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        # Predict on same data
        corrections = calibrator.predict(
            fragment_mz=sample_matches["fragment_mz"].values,
            rt=sample_matches["rt"].values,
            tic=sample_matches["tic"].values,
            intensity=sample_matches["observed_intensity"].values,
        )

        assert len(corrections) == len(sample_matches)
        assert isinstance(corrections, np.ndarray)

    def test_calibrator_predict_before_fit(self):
        """Test that predict fails before fit."""
        calibrator = MzCalibrator()

        with pytest.raises(ValueError, match="Model not trained"):
            calibrator.predict(
                fragment_mz=np.array([500.0]),
                rt=np.array([10.0]),
                tic=np.array([1e7]),
                intensity=np.array([1000.0]),
            )

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
        test_input = {
            "fragment_mz": np.array([500.0, 600.0]),
            "rt": np.array([10.0, 15.0]),
            "tic": np.array([1e7, 1e7]),
            "intensity": np.array([1000.0, 2000.0]),
        }

        orig_pred = calibrator.predict(**test_input)
        loaded_pred = loaded.predict(**test_input)
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_create_calibration_function(self, sample_matches):
        """Test calibration function creation."""
        calibrator = MzCalibrator(n_estimators=10)
        calibrator.fit(sample_matches)

        cal_func = calibrator.create_calibration_function()

        # Test with sample data
        metadata = {"rt": 10.0, "tic": 1e7}
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
