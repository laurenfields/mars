"""XGBoost-based m/z calibration model.

Learns m/z corrections based on:
- Precursor m/z (DIA isolation window)
- Fragment m/z
- Fragment charge
- Retention time
- Spectrum TIC (for space charge effects)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MzCalibrator:
    """XGBoost-based m/z calibration model.

    Predicts the expected m/z shift (delta_mz) based on spectral features,
    which can then be used to correct observed m/z values.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize calibrator.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (shrinkage)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names = [
            "fragment_mz",
            "rt",
            "log_tic",
            "log_intensity",
        ]
        self.training_stats: dict[str, Any] = {}

    def _prepare_features(
        self,
        fragment_mz: np.ndarray,
        rt: np.ndarray,
        tic: np.ndarray,
        intensity: np.ndarray,
    ) -> np.ndarray:
        """Prepare feature matrix for model.

        Args:
            fragment_mz: Fragment m/z values
            rt: Retention times
            tic: Total ion currents (spectrum level)
            intensity: Peak intensities (fragment level)

        Returns:
            Feature matrix (n_samples, 4)
        """
        # Log-transform TIC and intensity for better scaling
        log_tic = np.log10(np.clip(tic, 1, None))
        log_intensity = np.log10(np.clip(intensity, 1, None))

        X = np.column_stack(
            [
                fragment_mz,
                rt,
                log_tic,
                log_intensity,
            ]
        )

        return X

    def fit(
        self,
        matches: pd.DataFrame,
        validation_split: float = 0.2,
        sample_weight_col: str | None = "observed_intensity",
    ) -> MzCalibrator:
        """Train calibration model on fragment matches.

        Args:
            matches: DataFrame from match_library_to_spectra with columns:
                     precursor_mz, fragment_mz, fragment_charge, rt, tic, delta_mz
            validation_split: Fraction of data for validation
            sample_weight_col: Column to use for sample weights (observed_intensity
                              recommended - more intense fragments give better calibration)

        Returns:
            Self for chaining
        """
        import xgboost as xgb

        logger.info(f"Training XGBoost calibration model on {len(matches)} matches")

        # Prepare features
        X = self._prepare_features(
            matches["fragment_mz"].values,
            matches["rt"].values,
            matches["tic"].values,
            matches["observed_intensity"].values,
        )
        y = matches["delta_mz"].values

        # Optional sample weights
        sample_weight = None
        if sample_weight_col and sample_weight_col in matches.columns:
            sample_weight = matches[sample_weight_col].values
            # Normalize weights
            sample_weight = sample_weight / sample_weight.mean()

        # Train/validation split
        if validation_split > 0:
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                X,
                y,
                sample_weight if sample_weight is not None else np.ones(len(y)),
                test_size=validation_split,
                random_state=self.random_state,
            )
        else:
            X_train, y_train, w_train = X, y, sample_weight

        # Create XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )

        # Train
        eval_set = [(X_val, y_val)] if validation_split > 0 else None
        self.model.fit(
            X_train,
            y_train,
            sample_weight=w_train if sample_weight is not None else None,
            eval_set=eval_set,
            verbose=False,
        )

        # Calculate training statistics
        train_pred = self.model.predict(X_train)
        train_residuals = y_train - train_pred

        self.training_stats = {
            "n_samples": len(matches),
            "n_train": len(X_train),
            "n_val": len(X_val) if validation_split > 0 else 0,
            "train_mae": float(np.mean(np.abs(train_residuals))),
            "train_rmse": float(np.sqrt(np.mean(train_residuals**2))),
            "train_mean_delta": float(np.mean(y_train)),
            "train_std_delta": float(np.std(y_train)),
        }

        if validation_split > 0:
            val_pred = self.model.predict(X_val)
            val_residuals = y_val - val_pred
            self.training_stats["val_mae"] = float(np.mean(np.abs(val_residuals)))
            self.training_stats["val_rmse"] = float(np.sqrt(np.mean(val_residuals**2)))

        # Feature importance
        importance = self.model.feature_importances_
        self.training_stats["feature_importance"] = dict(
            zip(self.feature_names, importance.tolist())
        )

        logger.info("Training complete:")
        logger.info(f"  Train MAE: {self.training_stats['train_mae']:.4f} Th")
        logger.info(f"  Train RMSE: {self.training_stats['train_rmse']:.4f} Th")
        if validation_split > 0:
            logger.info(f"  Val MAE:   {self.training_stats['val_mae']:.4f} Th")
            logger.info(f"  Val RMSE:  {self.training_stats['val_rmse']:.4f} Th")

        logger.info("Feature importance:")
        for name, imp in self.training_stats["feature_importance"].items():
            logger.info(f"  {name}: {imp:.3f}")

        return self

    def predict(
        self,
        fragment_mz: np.ndarray,
        rt: np.ndarray,
        tic: np.ndarray,
        intensity: np.ndarray,
    ) -> np.ndarray:
        """Predict m/z correction for given features.

        Args:
            fragment_mz: Fragment m/z values
            rt: Retention times
            tic: Total ion currents (spectrum level)
            intensity: Peak intensities (fragment level)

        Returns:
            Predicted m/z corrections (subtract from observed m/z to recalibrate)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._prepare_features(
            fragment_mz,
            rt,
            tic,
            intensity,
        )

        return self.model.predict(X)

    def create_calibration_function(self):
        """Create a calibration function for use with write_calibrated_mzml.

        Returns:
            Function that takes (metadata, mz_array, intensity_array) and returns calibrated mz_array
        """

        def calibrate(
            metadata: dict, mz_array: np.ndarray, intensity_array: np.ndarray | None = None
        ) -> np.ndarray:
            """Apply calibration to an m/z array.

            Args:
                metadata: Dict with rt, precursor_mz, tic
                mz_array: Array of m/z values to calibrate
                intensity_array: Array of peak intensities (if None, uses default)

            Returns:
                Calibrated m/z array
            """
            if len(mz_array) == 0:
                return mz_array

            n = len(mz_array)

            # Use intensity array if provided, otherwise use a default
            if intensity_array is None or len(intensity_array) != n:
                intensity_array = np.full(n, 1000.0)  # Default moderate intensity

            # Get corrections for each m/z
            corrections = self.predict(
                fragment_mz=mz_array,
                rt=np.full(n, metadata.get("rt", 0.0)),
                tic=np.full(n, metadata.get("tic", 1e6)),
                intensity=intensity_array,
            )

            # Apply correction (subtract predicted delta to recalibrate)
            return mz_array - corrections

        return calibrate

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: Output file path (pickle format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "random_state": self.random_state,
                    "training_stats": self.training_stats,
                    "feature_names": self.feature_names,
                },
                f,
            )

        logger.info(f"Saved calibration model to {path}")

    @classmethod
    def load(cls, path: Path | str) -> MzCalibrator:
        """Load model from disk.

        Args:
            path: Model file path

        Returns:
            Loaded MzCalibrator
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        calibrator = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            learning_rate=data["learning_rate"],
            random_state=data["random_state"],
        )
        calibrator.model = data["model"]
        calibrator.training_stats = data["training_stats"]
        calibrator.feature_names = data.get("feature_names", calibrator.feature_names)

        logger.info(f"Loaded calibration model from {path}")
        return calibrator

    def get_stats_summary(self) -> str:
        """Get human-readable summary of training statistics.

        Returns:
            Formatted string with model statistics
        """
        if not self.training_stats:
            return "Model not trained"

        lines = [
            "Calibration Model Summary",
            "=" * 40,
            f"Training samples: {self.training_stats['n_samples']:,}",
            f"Train/Val split: {self.training_stats['n_train']:,} / {self.training_stats['n_val']:,}",
            "",
            "Original delta m/z:",
            f"  Mean: {self.training_stats['train_mean_delta']:.4f} Th",
            f"  Std:  {self.training_stats['train_std_delta']:.4f} Th",
            "",
            "Model performance:",
            f"  Train MAE:  {self.training_stats['train_mae']:.4f} Th",
            f"  Train RMSE: {self.training_stats['train_rmse']:.4f} Th",
        ]

        if "val_mae" in self.training_stats:
            lines.extend(
                [
                    f"  Val MAE:    {self.training_stats['val_mae']:.4f} Th",
                    f"  Val RMSE:   {self.training_stats['val_rmse']:.4f} Th",
                ]
            )

        lines.extend(["", "Feature importance:"])
        for name, imp in self.training_stats.get("feature_importance", {}).items():
            lines.append(f"  {name}: {imp:.3f}")

        return "\n".join(lines)
