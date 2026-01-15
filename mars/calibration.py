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
            "precursor_mz",
            "fragment_mz",
            "absolute_time",
            "log_tic",
            "log_intensity",
            "injection_time",
            "tic_injection_time",
            "fragment_ions",
            "ions_above_0_1",
            "ions_above_1_2",
            "ions_above_2_3",
            "adjacent_ratio_0_1",
            "adjacent_ratio_1_2",
            "adjacent_ratio_2_3",
            "rfa2_temp",
            "rfc2_temp",
        ]
        self.training_stats: dict[str, Any] = {}

    def _prepare_features(
        self,
        matches: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Prepare feature matrix for model, handling missing values and dynamic feature selection.

        Strategies for missing values:
        - If injection_time is missing in ALL rows: remove injection_time and tic_injection_time from features
        - If injection_time is missing in SOME rows: drop those rows with NaN values
        - If absolute_time is None: use fallback or skip if not available

        Args:
            matches: DataFrame from match_library_to_spectra with columns including:
                     precursor_mz, fragment_mz, absolute_time, log_tic, log_intensity,
                     injection_time, tic_injection_time

        Returns:
            Tuple of (feature_matrix, active_feature_names, filtered_df) where active_feature_names
            may be a subset of self.feature_names if some features are unavailable,
            and filtered_df is the DataFrame after dropping rows with NaN values
        """
        df = matches.copy()

        # Check which features are actually available
        active_features = []
        feature_data = {}

        # Always include these (should always be available)
        for col in ["precursor_mz", "fragment_mz", "log_tic", "log_intensity"]:
            if col in df.columns:
                active_features.append(col)
                feature_data[col] = df[col].values
            else:
                logger.warning(f"Required feature '{col}' not found in matches DataFrame")

        # Conditionally include optional features
        if "absolute_time" in df.columns and df["absolute_time"].notna().any():
            active_features.append("absolute_time")
            feature_data["absolute_time"] = df["absolute_time"].values

        # Check injection_time availability
        injection_time_available = (
            "injection_time" in df.columns and df["injection_time"].notna().any()
        )
        tic_injection_time_available = (
            "tic_injection_time" in df.columns and df["tic_injection_time"].notna().any()
        )

        if injection_time_available:
            # If injection_time is sparse (some missing), drop rows with NaN
            if df["injection_time"].isna().any():
                n_before = len(df)
                df = df.dropna(subset=["injection_time"])
                n_after = len(df)
                logger.info(
                    f"Dropped {n_before - n_after} rows with missing injection_time "
                    f"({n_after} rows retained)"
                )
                # Update feature_data for already-added features after row filtering
                for feat in active_features:
                    feature_data[feat] = df[feat].values

            active_features.append("injection_time")
            feature_data["injection_time"] = df["injection_time"].values

            if tic_injection_time_available:
                # If tic_injection_time sparse, drop additional rows
                if df["tic_injection_time"].isna().any():
                    n_before = len(df)
                    df = df.dropna(subset=["tic_injection_time"])
                    n_after = len(df)
                    logger.info(
                        f"Dropped {n_before - n_after} rows with missing tic_injection_time "
                        f"({n_after} rows retained)"
                    )

                active_features.append("tic_injection_time")
                feature_data["tic_injection_time"] = df["tic_injection_time"].values

            # Check fragment_ions availability (fragment_intensity × injection_time)
            fragment_ions_available = (
                "fragment_ions" in df.columns and df["fragment_ions"].notna().any()
            )
            if fragment_ions_available:
                if df["fragment_ions"].isna().any():
                    n_before = len(df)
                    df = df.dropna(subset=["fragment_ions"])
                    n_after = len(df)
                    logger.info(
                        f"Dropped {n_before - n_after} rows with missing fragment_ions "
                        f"({n_after} rows retained)"
                    )
                    # Update feature_data for already-added features after row filtering
                    for feat in active_features:
                        feature_data[feat] = df[feat].values

                active_features.append("fragment_ions")
                feature_data["fragment_ions"] = df["fragment_ions"].values

            # Check adjacent ion population features (only if injection_time is available)
            for ions_col in [
                "ions_above_0_1",
                "ions_above_1_2",
                "ions_above_2_3",
                "ions_below_0_1",
                "ions_below_1_2",
                "ions_below_2_3",
            ]:
                if ions_col in df.columns and df[ions_col].notna().any():
                    if df[ions_col].isna().any():
                        n_before = len(df)
                        df = df.dropna(subset=[ions_col])
                        n_after = len(df)
                        logger.info(
                            f"Dropped {n_before - n_after} rows with missing {ions_col} "
                            f"({n_after} rows retained)"
                        )
                        # Update feature_data for already-added features after row filtering
                        for feat in active_features:
                            feature_data[feat] = df[feat].values

                    active_features.append(ions_col)
                    feature_data[ions_col] = df[ions_col].values

            # Check adjacent ratio features (only if fragment_ions is available)
            for ratio_col in [
                "adjacent_ratio_0_1",
                "adjacent_ratio_1_2",
                "adjacent_ratio_2_3",
                "adjacent_ratio_below_0_1",
                "adjacent_ratio_below_1_2",
                "adjacent_ratio_below_2_3",
            ]:
                if ratio_col in df.columns and df[ratio_col].notna().any():
                    if df[ratio_col].isna().any():
                        n_before = len(df)
                        df = df.dropna(subset=[ratio_col])
                        n_after = len(df)
                        logger.info(
                            f"Dropped {n_before - n_after} rows with missing {ratio_col} "
                            f"({n_after} rows retained)"
                        )
                        # Update feature_data for already-added features after row filtering
                        for feat in active_features:
                            feature_data[feat] = df[feat].values

                    active_features.append(ratio_col)
                    feature_data[ratio_col] = df[ratio_col].values
        else:
            logger.info(
                "injection_time not available in all spectra - skipping injection_time features"
            )

        # Check temperature features availability
        for temp_col in ["rfa2_temp", "rfc2_temp"]:
            if temp_col in df.columns and df[temp_col].notna().any():
                # If temperature is sparse, drop rows with NaN
                if df[temp_col].isna().any():
                    n_before = len(df)
                    df = df.dropna(subset=[temp_col])
                    n_after = len(df)
                    logger.info(
                        f"Dropped {n_before - n_after} rows with missing {temp_col} "
                        f"({n_after} rows retained)"
                    )
                    # Update feature_data for already-added features after row filtering
                    for feat in active_features:
                        feature_data[feat] = df[feat].values

                active_features.append(temp_col)
                feature_data[temp_col] = df[temp_col].values
            else:
                logger.info(f"{temp_col} not available - skipping this temperature feature")

        # Build feature matrix with active features only
        X = np.column_stack([feature_data[col] for col in active_features])

        # Update self.feature_names to reflect what was actually used
        self.feature_names = active_features

        logger.info(f"Using {len(active_features)} features: {active_features}")

        return X, active_features, df

    def fit(
        self,
        matches: pd.DataFrame,
        validation_split: float = 0.2,
        sample_weight_col: str | None = "observed_intensity",
    ) -> MzCalibrator:
        """Train calibration model on fragment matches.

        Args:
            matches: DataFrame from match_library_to_spectra with columns:
                     precursor_mz, fragment_mz, absolute_time, log_tic, log_intensity,
                     injection_time, tic_injection_time, delta_mz
            validation_split: Fraction of data for validation
            sample_weight_col: Column to use for sample weights (observed_intensity
                              recommended - more intense fragments give better calibration)

        Returns:
            Self for chaining
        """
        import xgboost as xgb

        logger.info(f"Training XGBoost calibration model on {len(matches)} matches")

        # Prepare features (this also updates self.feature_names based on availability)
        X, active_features, filtered_matches = self._prepare_features(matches)
        y = filtered_matches["delta_mz"].values

        # Optional sample weights
        sample_weight = None
        if sample_weight_col and sample_weight_col in filtered_matches.columns:
            sample_weight = filtered_matches[sample_weight_col].values
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
            zip(active_features, importance.tolist(), strict=True)
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
        matches: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict m/z correction for given features.

        Can be called in two ways:
        1. predict(matches_df) - DataFrame with all feature columns
        2. predict(feature1=arr1, feature2=arr2, ...) - explicit feature arrays (legacy, only if
           all features match current model exactly)

        Args:
            matches: DataFrame with feature columns matching active features, or None
            **kwargs: Named feature arrays (only if matches is None and using legacy interface)

        Returns:
            Predicted m/z corrections (subtract from observed m/z to recalibrate)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if matches is not None:
            # Use DataFrame interface
            X, _, _ = self._prepare_features(matches)
        else:
            # Legacy interface: build feature matrix from kwargs
            if not self.feature_names:
                raise ValueError("No features defined for prediction")

            feature_data = {}
            for feat in self.feature_names:
                if feat not in kwargs:
                    raise ValueError(f"Missing required feature for prediction: {feat}")
                feature_data[feat] = kwargs[feat]

            X = np.column_stack([feature_data[col] for col in self.feature_names])

        return self.model.predict(X)

    def create_calibration_function(self):
        """Create a calibration function for use with write_calibrated_mzml.

        Returns:
            Function that takes (metadata, mz_array, intensity_array) and returns calibrated mz_array
        """

        def _compute_ions_in_range_vectorized(
            mz_array: np.ndarray,
            cumsum: np.ndarray,
            offset_low: float,
            offset_high: float,
        ) -> np.ndarray:
            """Compute sum of intensities in m/z range for each peak (fully vectorized).

            For each m/z value x[i], computes sum of intensities in range
            (x[i] + offset_low, x[i] + offset_high].

            Args:
                mz_array: Sorted array of m/z values
                cumsum: Cumulative sum of intensities (prepended with 0)
                offset_low: Lower offset from m/z (exclusive)
                offset_high: Upper offset from m/z (inclusive)

            Returns:
                Array of intensity sums for each peak
            """
            # Fully vectorized: searchsorted can take arrays as second argument
            low_mz = mz_array + offset_low
            high_mz = mz_array + offset_high
            low_idx = np.searchsorted(mz_array, low_mz, side="right")
            high_idx = np.searchsorted(mz_array, high_mz, side="right")
            return cumsum[high_idx] - cumsum[low_idx]

        def calibrate(
            metadata: dict, mz_array: np.ndarray, intensity_array: np.ndarray | None = None
        ) -> np.ndarray:
            """Apply calibration to an m/z array.

            Args:
                metadata: Dict with feature values. Required keys depend on active features
                          Typical keys: precursor_mz, absolute_time, tic, injection_time
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

            injection_time = metadata.get("injection_time", 0.0)

            # Check if we need space charge features
            space_charge_features = {
                "ions_above_0_1",
                "ions_above_1_2",
                "ions_above_2_3",
                "ions_below_0_1",
                "ions_below_1_2",
                "ions_below_2_3",
                "adjacent_ratio_0_1",
                "adjacent_ratio_1_2",
                "adjacent_ratio_2_3",
                "adjacent_ratio_below_0_1",
                "adjacent_ratio_below_1_2",
                "adjacent_ratio_below_2_3",
            }
            need_space_charge = bool(space_charge_features & set(self.feature_names))

            # Pre-compute space charge features if needed
            space_charge_data = {}
            if need_space_charge and injection_time > 0:
                # Build cumulative sum for efficient range queries
                cumsum = np.zeros(n + 1)
                cumsum[1:] = np.cumsum(intensity_array)

                # Compute ions above features (fully vectorized)
                ions_above_0_1 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, 0.5, 1.5)
                    * injection_time
                )
                ions_above_1_2 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, 1.5, 2.5)
                    * injection_time
                )
                ions_above_2_3 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, 2.5, 3.5)
                    * injection_time
                )

                # Compute ions below features (fully vectorized)
                ions_below_0_1 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, -1.5, -0.5)
                    * injection_time
                )
                ions_below_1_2 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, -2.5, -1.5)
                    * injection_time
                )
                ions_below_2_3 = (
                    _compute_ions_in_range_vectorized(mz_array, cumsum, -3.5, -2.5)
                    * injection_time
                )

                # Store in dict
                space_charge_data["ions_above_0_1"] = ions_above_0_1
                space_charge_data["ions_above_1_2"] = ions_above_1_2
                space_charge_data["ions_above_2_3"] = ions_above_2_3
                space_charge_data["ions_below_0_1"] = ions_below_0_1
                space_charge_data["ions_below_1_2"] = ions_below_1_2
                space_charge_data["ions_below_2_3"] = ions_below_2_3

                # Compute fragment_ions for ratio calculations
                fragment_ions = intensity_array * injection_time

                # Compute ratio features (avoid division by zero)
                with np.errstate(divide="ignore", invalid="ignore"):
                    space_charge_data["adjacent_ratio_0_1"] = np.where(
                        fragment_ions > 0, ions_above_0_1 / fragment_ions, 0.0
                    )
                    space_charge_data["adjacent_ratio_1_2"] = np.where(
                        fragment_ions > 0, ions_above_1_2 / fragment_ions, 0.0
                    )
                    space_charge_data["adjacent_ratio_2_3"] = np.where(
                        fragment_ions > 0, ions_above_2_3 / fragment_ions, 0.0
                    )
                    space_charge_data["adjacent_ratio_below_0_1"] = np.where(
                        fragment_ions > 0, ions_below_0_1 / fragment_ions, 0.0
                    )
                    space_charge_data["adjacent_ratio_below_1_2"] = np.where(
                        fragment_ions > 0, ions_below_1_2 / fragment_ions, 0.0
                    )
                    space_charge_data["adjacent_ratio_below_2_3"] = np.where(
                        fragment_ions > 0, ions_below_2_3 / fragment_ions, 0.0
                    )

            # Build feature arrays for prediction
            feature_data = {}

            for feat in self.feature_names:
                if feat == "precursor_mz":
                    feature_data[feat] = np.full(n, metadata.get("precursor_mz", 0.0))
                elif feat == "fragment_mz":
                    feature_data[feat] = mz_array
                elif feat == "log_tic":
                    tic = metadata.get("tic", 1e6)
                    feature_data[feat] = np.full(n, np.log10(np.clip(tic, 1, None)))
                elif feat == "log_intensity":
                    feature_data[feat] = np.log10(np.clip(intensity_array, 1, None))
                elif feat == "absolute_time":
                    feature_data[feat] = np.full(n, metadata.get("absolute_time", 0.0))
                elif feat == "injection_time":
                    feature_data[feat] = np.full(n, injection_time)
                elif feat == "tic_injection_time":
                    tic = metadata.get("tic", 1e6)
                    feature_data[feat] = np.full(n, tic * injection_time)
                elif feat == "fragment_ions":
                    # fragment_ions = intensity × injection_time
                    feature_data[feat] = intensity_array * injection_time
                elif feat == "rfa2_temp":
                    feature_data[feat] = np.full(n, metadata.get("rfa2_temp", 0.0))
                elif feat == "rfc2_temp":
                    feature_data[feat] = np.full(n, metadata.get("rfc2_temp", 0.0))
                elif feat in space_charge_data:
                    # Use pre-computed space charge features
                    feature_data[feat] = space_charge_data[feat]
                else:
                    # Unknown feature - use default of 0
                    feature_data[feat] = np.full(n, 0.0)

            # Build feature matrix in correct order
            X = np.column_stack([feature_data[col] for col in self.feature_names])

            # Get corrections
            corrections = self.model.predict(X)

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
