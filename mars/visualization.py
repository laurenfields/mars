"""QC visualization for mass calibration assessment.

Generates before/after comparison plots:
1. Delta m/z histogram: Distribution of mass errors
2. Delta m/z heatmap: 2D visualization (RT x fragment m/z)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_delta_mz_histogram(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Delta m/z Distribution",
    bins: int = 200,
    xlim: tuple[float, float] | None = None,
    intensity_weighted: bool = False,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot histogram of delta m/z distribution, before and after calibration.

    Args:
        before: DataFrame with 'delta_mz', 'delta_ppm', 'observed_intensity' columns
        after: Optional DataFrame with 'delta_mz_calibrated' or 'delta_ppm_calibrated' column
        output_path: Path to save figure (optional)
        title: Plot title
        bins: Number of histogram bins
        xlim: X-axis limits (defaults: ±0.5 Th or ±30 ppm)
        intensity_weighted: If True, use intensity-weighted histogram
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    # Set default xlim based on mode
    if xlim is None:
        xlim = (-30.0, 30.0) if use_ppm else (-0.5, 0.5)

    # Column names based on mode
    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"
    fmt = ".2f" if use_ppm else ".4f"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(7, 5))
        ax_after = None

    def compute_robust_stats(delta: np.ndarray, weights: np.ndarray | None = None):
        """Compute robust statistics."""
        median = np.median(delta)
        mad = np.median(np.abs(delta - median))  # Median Absolute Deviation
        p25, p75 = np.percentile(delta, [25, 75])
        iqr = p75 - p25

        # Intensity-weighted mean if weights provided
        if weights is not None and len(weights) == len(delta):
            weights = weights / weights.sum()
            wmean = np.sum(delta * weights)
        else:
            wmean = np.mean(delta)

        # Root mean square error
        rms = np.sqrt(np.mean(delta**2))

        return {
            "median": median,
            "mad": mad,
            "iqr": iqr,
            "p25": p25,
            "p75": p75,
            "wmean": wmean,
            "std": np.std(delta),
            "rms": rms,
        }

    def make_histogram(
        df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str, color: str
    ):
        """Create histogram with robust statistics."""
        delta = df[delta_col].dropna().values
        weights = df["observed_intensity"].values if "observed_intensity" in df.columns else None

        if intensity_weighted and weights is not None:
            # Intensity-weighted histogram
            ax.hist(
                delta,
                bins=bins,
                range=xlim,
                weights=weights / weights.max(),  # Normalize weights for display
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_ylabel("Intensity-Weighted Count", fontsize=12)
        else:
            ax.hist(
                delta,
                bins=bins,
                range=xlim,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_ylabel("Count", fontsize=12)

        stats = compute_robust_stats(delta, weights)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"0 {unit}")
        ax.axvline(
            stats["median"],
            color="orange",
            linestyle="-",
            linewidth=1.5,
            label=f"Median: {stats['median']:{fmt}} {unit}",
        )

        ax.set_xlabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_title(subplot_title, fontsize=14)
        ax.legend(loc="upper right")
        ax.set_xlim(xlim)

        # Robust statistics text box
        stats_text = (
            f"n = {len(delta):,}\n"
            f"Median = {stats['median']:{fmt}}\n"
            f"MAD = {stats['mad']:{fmt}}\n"
            f"RMS = {stats['rms']:{fmt}}\n"
            f"Wt.Mean = {stats['wmean']:{fmt}}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        return stats

    # Before calibration
    stats_before = make_histogram(
        before, ax_before, delta_col_before, "Before Calibration", "steelblue"
    )

    # After calibration
    stats_after = None
    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        stats_after = make_histogram(
            after, ax_after, actual_delta_col, "After Calibration", "forestgreen"
        )

        # Add improvement summary
        mad_improvement = (
            (1 - stats_after["mad"] / stats_before["mad"]) * 100 if stats_before["mad"] > 0 else 0
        )
        rms_improvement = (
            (1 - stats_after["rms"] / stats_before["rms"]) * 100 if stats_before["rms"] > 0 else 0
        )
        fig.suptitle(
            f"{title}\nMAD improved by {mad_improvement:.1f}%, RMS improved by {rms_improvement:.1f}%",
            fontsize=14,
            fontweight="bold",
        )
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved histogram to {output_path}")

    return fig


def plot_delta_mz_heatmap(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Delta m/z Heatmap (RT x Fragment m/z)",
    rt_bins: int = 50,
    mz_bins: int = 50,
    vmin: float | None = None,
    vmax: float | None = None,
    vlim: tuple[float, float] | None = None,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D heatmap of delta m/z (X=RT, Y=fragment m/z, Color=median delta).

    Args:
        before: DataFrame with 'rt', 'fragment_mz', 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        rt_bins: Number of RT bins
        mz_bins: Number of m/z bins
        vmin: Color scale minimum (deprecated, use vlim)
        vmax: Color scale maximum (deprecated, use vlim)
        vlim: Color scale limits as (vmin, vmax)
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    # Handle vlim parameter
    if vlim is not None:
        vmin_val, vmax_val = vlim
    elif vmin is not None and vmax is not None:
        vmin_val, vmax_val = vmin, vmax
    else:
        vmin_val, vmax_val = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    # Column names based on mode
    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_heatmap(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        """Create binned heatmap."""
        # Use absolute_time if available, otherwise use rt
        time_col = "absolute_time" if "absolute_time" in df.columns else "rt"

        # Create bins
        rt_edges = np.linspace(df[time_col].min(), df[time_col].max(), rt_bins + 1)
        mz_edges = np.linspace(df["fragment_mz"].min(), df["fragment_mz"].max(), mz_bins + 1)

        # Assign bins
        df = df.copy()
        df["rt_bin"] = pd.cut(df[time_col], bins=rt_edges, labels=False, include_lowest=True)
        df["mz_bin"] = pd.cut(df["fragment_mz"], bins=mz_edges, labels=False, include_lowest=True)

        # Aggregate by bins (median delta m/z)
        heatmap_data = (
            df.groupby(["mz_bin", "rt_bin"])[delta_col].median().unstack(fill_value=np.nan)
        )

        # Plot
        im = ax.imshow(
            heatmap_data.values,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin_val,
            vmax=vmax_val,
            extent=[rt_edges[0], rt_edges[-1], mz_edges[0], mz_edges[-1]],
        )

        ax.set_xlabel(
            "Acquisition Time (s)" if time_col == "absolute_time" else "Retention Time (min)",
            fontsize=12,
        )
        ax.set_ylabel("Fragment m/z (Th)", fontsize=12)
        ax.set_title(subplot_title, fontsize=12)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f"Median Delta m/z ({unit})", fontsize=10)

        return heatmap_data

    # Before calibration
    make_heatmap(before, ax_before, delta_col_before, "Before Calibration")

    # After calibration
    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_heatmap(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved heatmap to {output_path}")

    return fig


def plot_feature_importance(
    calibrator,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot feature importance from calibration model.

    Args:
        calibrator: Trained MzCalibrator
        output_path: Path to save figure

    Returns:
        Matplotlib Figure
    """
    importance = calibrator.training_stats.get("feature_importance", {})
    if not importance:
        raise ValueError("No feature importance in model. Train model first.")

    # Map feature names to display names
    display_name_map = {
        "ions_above_0_1": "ions_above_0.5_1.5",
        "ions_above_1_2": "ions_above_1.5_2.5",
        "ions_above_2_3": "ions_above_2.5_3.5",
        "ions_below_0_1": "ions_below_-1.5_-0.5",
        "ions_below_1_2": "ions_below_-2.5_-1.5",
        "ions_below_2_3": "ions_below_-3.5_-2.5",
        "adjacent_ratio_0_1": "adjacent_ratio_0.5_1.5",
        "adjacent_ratio_1_2": "adjacent_ratio_1.5_2.5",
        "adjacent_ratio_2_3": "adjacent_ratio_2.5_3.5",
        "adjacent_ratio_below_0_1": "adjacent_ratio_-1.5_-0.5",
        "adjacent_ratio_below_1_2": "adjacent_ratio_-2.5_-1.5",
        "adjacent_ratio_below_2_3": "adjacent_ratio_-3.5_-2.5",
    }

    features = list(importance.keys())
    values = list(importance.values())

    # Replace feature names with display names
    features = [display_name_map.get(f, f) for f in features]

    # Sort by importance
    sorted_idx = np.argsort(values)
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    ax.barh(features, values, color=colors)

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Calibration Model Feature Importance", fontsize=14, fontweight="bold")

    # Add value labels
    for i, (_f, v) in enumerate(zip(features, values, strict=True)):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {output_path}")

    return fig


def plot_intensity_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Intensity vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of log10 intensity vs delta m/z.

    Args:
        before: DataFrame with 'observed_intensity' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error (defaults: ±0.25 Th or ±25 ppm)
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    # Set default ylim based on mode
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    # Column names based on mode
    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        x = np.log10(np.clip(df["observed_intensity"], 1, None))
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Intensity)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved intensity vs error plot to {output_path}")

    return fig


def plot_rt_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "RT vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of retention time vs delta m/z.

    Args:
        before: DataFrame with 'rt' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        # Use absolute_time if available, otherwise use rt
        time_col = "absolute_time" if "absolute_time" in df.columns else "rt"
        x = df[time_col].values
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(
            "Acquisition Time (s)" if time_col == "absolute_time" else "Retention Time (min)",
            fontsize=12,
        )
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved RT vs error plot to {output_path}")

    return fig


def plot_fragment_mz_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Fragment m/z vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of fragment m/z vs delta m/z.

    Args:
        before: DataFrame with 'fragment_mz' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        x = df["fragment_mz"].values
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Fragment m/z (Th)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved fragment m/z vs error plot to {output_path}")

    return fig


def plot_tic_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Spectrum TIC vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of log10 spectrum TIC vs delta m/z.

    Args:
        before: DataFrame with 'tic' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        # Use log_tic if available, otherwise calculate from tic
        if "log_tic" in df.columns:
            x = df["log_tic"].values
        else:
            x = np.log10(np.clip(df["tic"], 1, None))
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Spectrum TIC)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved TIC vs error plot to {output_path}")

    return fig


def plot_injection_time_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Injection Time vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of injection time vs delta m/z.

    Args:
        before: DataFrame with 'injection_time' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        # Use injection_time if available
        if "injection_time" not in df.columns or df["injection_time"].isna().all():
            ax.text(
                0.5,
                0.5,
                "No injection time data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        # Filter out NaN values
        mask = ~df["injection_time"].isna() & ~df[delta_col].isna()
        x = df.loc[mask, "injection_time"].values
        y = df.loc[mask, delta_col].values

        # Calculate data range for x-axis
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        padding = x_range * 0.05 if x_range > 0 else 0.001

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Ion Injection Time (s)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Log10(Fragment count)")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved injection time vs error plot to {output_path}")

    return fig


def plot_fragment_ions_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "Fragment Ions vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of log10 fragment ions vs delta m/z.

    Args:
        before: DataFrame with 'fragment_ions' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        # Use fragment_ions if available
        if "fragment_ions" not in df.columns or df["fragment_ions"].isna().all():
            ax.text(
                0.5,
                0.5,
                "No fragment ions data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        # Filter out NaN and zero values
        mask = ~df["fragment_ions"].isna() & ~df[delta_col].isna() & (df["fragment_ions"] > 0)
        if not mask.any():
            ax.text(
                0.5,
                0.5,
                "No valid fragment ions data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        x = np.log10(df.loc[mask, "fragment_ions"].values)
        y = df.loc[mask, delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Fragment Ions)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved fragment ions vs error plot to {output_path}")

    return fig


def plot_tic_injection_time_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    title: str = "TIC×Injection Time vs Mass Error",
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of TIC×injection_time vs delta m/z.

    Args:
        before: DataFrame with 'tic_injection_time' and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        # Use tic_injection_time if available
        if "tic_injection_time" not in df.columns or df["tic_injection_time"].isna().all():
            ax.text(
                0.5,
                0.5,
                "No TIC×injection time data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        # Filter out NaN values
        mask = ~df["tic_injection_time"].isna() & ~df[delta_col].isna()
        x_raw = df.loc[mask, "tic_injection_time"].values
        x = np.log10(np.clip(x_raw, 1, None))
        y = df.loc[mask, delta_col].values

        # Calculate data range for x-axis
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        padding = x_range * 0.05 if x_range > 0 else 0.05

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(TIC×Injection Time)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, delta_col_before, "Before Calibration")

    if ax_after is not None and after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, ax_after, actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved TIC×injection time vs error plot to {output_path}")

    return fig


def plot_single_temperature_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    temp_col: str = "rfa2_temp",
    temp_label: str = "RFA2 (RF Amplifier)",
    output_path: Path | str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of a single temperature feature vs delta m/z.

    Args:
        before: DataFrame with temperature column and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        temp_col: Column name for temperature (e.g., 'rfa2_temp', 'rfc2_temp')
        temp_label: Human-readable label for the temperature
        output_path: Path to save figure
        title: Plot title (defaults to "{temp_label} vs Mass Error")
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if title is None:
        title = f"{temp_label} vs Mass Error"

    # Check if temperature data is available
    has_temp = temp_col in before.columns and before[temp_col].notna().any()

    if not has_temp:
        # No temperature data - create empty figure with message
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"No {temp_label} data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Create subplot layout
    n_cols = 2 if after is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        mask = ~df[temp_col].isna() & ~df[delta_col].isna()
        x = df.loc[mask, temp_col].values
        y = df.loc[mask, delta_col].values

        if len(x) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        # Calculate data range for x-axis
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        padding = x_range * 0.05 if x_range > 0 else 0.5

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(f"{temp_label} Temperature (°C)", fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    # Before calibration
    make_hexbin(before, axes[0, 0], delta_col_before, "Before Calibration")

    # After calibration
    if after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, axes[0, 1], actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {temp_label} temperature vs error plot to {output_path}")

    return fig


def plot_adjacent_ion_feature_vs_error(
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
    feature_col: str = "ions_above_0_1",
    feature_label: str = "Ions Above (0.5-1.5 Th)",
    output_path: Path | str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    gridsize: int = 100,
    use_log_scale: bool = True,
    use_ppm: bool = False,
) -> plt.Figure:
    """Plot 2D hexbin of an adjacent ion feature vs delta m/z.

    Args:
        before: DataFrame with feature column and 'delta_mz'/'delta_ppm' columns
        after: Optional DataFrame with 'delta_mz_calibrated'/'delta_ppm_calibrated' column
        feature_col: Column name for the feature (e.g., 'ions_above_0_1', 'adjacent_ratio_0_1')
        feature_label: Human-readable label for the feature
        output_path: Path to save figure
        title: Plot title (defaults to "{feature_label} vs Mass Error")
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size
        use_log_scale: If True, use log10 scale for the x-axis (for ions_above features)
        use_ppm: If True, plot in ppm units instead of Th

    Returns:
        Matplotlib Figure
    """
    if ylim is None:
        ylim = (-25.0, 25.0) if use_ppm else (-0.25, 0.25)

    delta_col_before = "delta_ppm" if use_ppm else "delta_mz"
    delta_col_after = "delta_ppm_calibrated" if use_ppm else "delta_mz_calibrated"
    unit = "ppm" if use_ppm else "Th"

    if title is None:
        title = f"{feature_label} vs Mass Error"

    # Check if feature data is available
    has_feature = feature_col in before.columns and before[feature_col].notna().any()

    if not has_feature:
        # No feature data - create empty figure with message
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"No {feature_label} data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Create subplot layout
    n_cols = 2 if after is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        mask = ~df[feature_col].isna() & ~df[delta_col].isna()

        # For log scale, also filter out zeros and negatives
        if use_log_scale:
            mask = mask & (df[feature_col] > 0)

        x_raw = df.loc[mask, feature_col].values
        y = df.loc[mask, delta_col].values

        if len(x_raw) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(subplot_title, fontsize=12)
            return

        # Apply log scale if requested
        if use_log_scale:
            x = np.log10(x_raw)
            x_label = f"Log10({feature_label})"
        else:
            x = x_raw
            x_label = feature_label

        # Calculate data range for x-axis
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        padding = x_range * 0.05 if x_range > 0 else 0.1

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"Delta m/z ({unit})", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    # Before calibration
    make_hexbin(before, axes[0, 0], delta_col_before, "Before Calibration")

    # After calibration
    if after is not None:
        actual_delta_col = delta_col_after if delta_col_after in after.columns else delta_col_before
        make_hexbin(after, axes[0, 1], actual_delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {feature_label} vs error plot to {output_path}")

    return fig


def generate_qc_report(
    before: pd.DataFrame,
    after: pd.DataFrame | None,
    calibrator,
    output_dir: Path | str,
    file_prefix: str = "mars_qc",
    use_ppm: bool | None = None,
) -> list[Path]:
    """Generate full QC report with all plots.

    Args:
        before: DataFrame with matches before calibration
        after: DataFrame with matches after calibration (with delta_mz_calibrated)
        calibrator: Trained MzCalibrator
        output_dir: Output directory for plots
        file_prefix: Prefix for output files
        use_ppm: If True, plot errors in ppm. If False, plot in Th.
                If None (default), auto-detect based on error magnitude.

    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # Auto-detect whether to use ppm or Th based on error spread (MAD)
    if use_ppm is None:
        # Use MAD (Median Absolute Deviation) to measure error spread
        # MAD = median(|delta_mz - median(delta_mz)|)
        median_error = before["delta_mz"].median()
        mad = (before["delta_mz"] - median_error).abs().median()
        # If MAD < 0.05 Th, likely high-resolution data -> use ppm
        use_ppm = mad < 0.05
        if use_ppm:
            logger.info(f"Auto-detected ppm mode (MAD = {mad:.4f} Th)")
        else:
            logger.info(f"Auto-detected Th mode (MAD = {mad:.4f} Th)")

    # Set appropriate axis limits based on mode
    if use_ppm:
        ylim: tuple[float, float] = (-25.0, 25.0)  # ±25 ppm for high-resolution data
    else:
        ylim = (-0.25, 0.25)  # ±0.25 Th for unit resolution data

    # Histogram
    hist_path = output_dir / f"{file_prefix}_histogram.png"
    plot_delta_mz_histogram(before, after, hist_path, use_ppm=use_ppm)
    generated_files.append(hist_path)
    plt.close()

    # Heatmap (RT x fragment m/z)
    heatmap_path = output_dir / f"{file_prefix}_heatmap.png"
    plot_delta_mz_heatmap(before, after, heatmap_path, use_ppm=use_ppm, vlim=ylim)
    generated_files.append(heatmap_path)
    plt.close()

    # Intensity vs error
    intensity_path = output_dir / f"{file_prefix}_intensity_vs_error.png"
    plot_intensity_vs_error(before, after, intensity_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(intensity_path)
    plt.close()

    # RT vs error
    rt_path = output_dir / f"{file_prefix}_rt_vs_error.png"
    plot_rt_vs_error(before, after, rt_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(rt_path)
    plt.close()

    # Fragment m/z vs error
    mz_path = output_dir / f"{file_prefix}_mz_vs_error.png"
    plot_fragment_mz_vs_error(before, after, mz_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(mz_path)
    plt.close()

    # TIC vs error
    tic_path = output_dir / f"{file_prefix}_tic_vs_error.png"
    plot_tic_vs_error(before, after, tic_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(tic_path)
    plt.close()

    # Injection time vs error
    injection_time_path = output_dir / f"{file_prefix}_injection_time_vs_error.png"
    plot_injection_time_vs_error(before, after, injection_time_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(injection_time_path)
    plt.close()

    # TIC×Injection time vs error
    tic_injection_time_path = output_dir / f"{file_prefix}_tic_injection_time_vs_error.png"
    plot_tic_injection_time_vs_error(
        before, after, tic_injection_time_path, use_ppm=use_ppm, ylim=ylim
    )
    generated_files.append(tic_injection_time_path)
    plt.close()

    # Fragment ions vs error
    fragment_ions_path = output_dir / f"{file_prefix}_fragment_ions_vs_error.png"
    plot_fragment_ions_vs_error(before, after, fragment_ions_path, use_ppm=use_ppm, ylim=ylim)
    generated_files.append(fragment_ions_path)
    plt.close()

    # RFA2 Temperature vs error (if available)
    if "rfa2_temp" in before.columns and before["rfa2_temp"].notna().any():
        rfa2_path = output_dir / f"{file_prefix}_rfa2_temperature_vs_error.png"
        plot_single_temperature_vs_error(
            before,
            after,
            temp_col="rfa2_temp",
            temp_label="RFA2 (RF Amplifier)",
            output_path=rfa2_path,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(rfa2_path)
        plt.close()

    # RFC2 Temperature vs error (if available)
    if "rfc2_temp" in before.columns and before["rfc2_temp"].notna().any():
        rfc2_path = output_dir / f"{file_prefix}_rfc2_temperature_vs_error.png"
        plot_single_temperature_vs_error(
            before,
            after,
            temp_col="rfc2_temp",
            temp_label="RFC2 (RF Electronics)",
            output_path=rfc2_path,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(rfc2_path)
        plt.close()

    # Adjacent ion feature plots (if available)
    # ions_above_0_1 vs error
    if "ions_above_0_1" in before.columns and before["ions_above_0_1"].notna().any():
        ions_0_1_path = output_dir / f"{file_prefix}_ions_above_0.5_1.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_above_0_1",
            feature_label="Ions Above (0.5-1.5 Th)",
            output_path=ions_0_1_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_0_1_path)
        plt.close()

    # ions_above_1_2 vs error
    if "ions_above_1_2" in before.columns and before["ions_above_1_2"].notna().any():
        ions_1_2_path = output_dir / f"{file_prefix}_ions_above_1.5_2.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_above_1_2",
            feature_label="Ions Above (1.5-2.5 Th)",
            output_path=ions_1_2_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_1_2_path)
        plt.close()

    # ions_above_2_3 vs error
    if "ions_above_2_3" in before.columns and before["ions_above_2_3"].notna().any():
        ions_2_3_path = output_dir / f"{file_prefix}_ions_above_2.5_3.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_above_2_3",
            feature_label="Ions Above (2.5-3.5 Th)",
            output_path=ions_2_3_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_2_3_path)
        plt.close()

    # adjacent_ratio_0_1 vs error
    if "adjacent_ratio_0_1" in before.columns and before["adjacent_ratio_0_1"].notna().any():
        ratio_0_1_path = output_dir / f"{file_prefix}_adjacent_ratio_0.5_1.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_0_1",
            feature_label="Adjacent Ratio (0.5-1.5 Th)",
            output_path=ratio_0_1_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_0_1_path)
        plt.close()

    # adjacent_ratio_1_2 vs error
    if "adjacent_ratio_1_2" in before.columns and before["adjacent_ratio_1_2"].notna().any():
        ratio_1_2_path = output_dir / f"{file_prefix}_adjacent_ratio_1.5_2.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_1_2",
            feature_label="Adjacent Ratio (1.5-2.5 Th)",
            output_path=ratio_1_2_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_1_2_path)
        plt.close()

    # adjacent_ratio_2_3 vs error
    if "adjacent_ratio_2_3" in before.columns and before["adjacent_ratio_2_3"].notna().any():
        ratio_2_3_path = output_dir / f"{file_prefix}_adjacent_ratio_2.5_3.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_2_3",
            feature_label="Adjacent Ratio (2.5-3.5 Th)",
            output_path=ratio_2_3_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_2_3_path)
        plt.close()

    # ions_below_0_1 vs error
    if "ions_below_0_1" in before.columns and before["ions_below_0_1"].notna().any():
        ions_below_0_1_path = output_dir / f"{file_prefix}_ions_below_-1.5_-0.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_below_0_1",
            feature_label="Ions Below (-1.5 to -0.5 Th)",
            output_path=ions_below_0_1_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_below_0_1_path)
        plt.close()

    # ions_below_1_2 vs error
    if "ions_below_1_2" in before.columns and before["ions_below_1_2"].notna().any():
        ions_below_1_2_path = output_dir / f"{file_prefix}_ions_below_-2.5_-1.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_below_1_2",
            feature_label="Ions Below (-2.5 to -1.5 Th)",
            output_path=ions_below_1_2_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_below_1_2_path)
        plt.close()

    # ions_below_2_3 vs error
    if "ions_below_2_3" in before.columns and before["ions_below_2_3"].notna().any():
        ions_below_2_3_path = output_dir / f"{file_prefix}_ions_below_-3.5_-2.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="ions_below_2_3",
            feature_label="Ions Below (-3.5 to -2.5 Th)",
            output_path=ions_below_2_3_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ions_below_2_3_path)
        plt.close()

    # adjacent_ratio_below_0_1 vs error
    if "adjacent_ratio_below_0_1" in before.columns and before["adjacent_ratio_below_0_1"].notna().any():
        ratio_below_0_1_path = output_dir / f"{file_prefix}_adjacent_ratio_-1.5_-0.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_below_0_1",
            feature_label="Adjacent Ratio (-1.5 to -0.5 Th)",
            output_path=ratio_below_0_1_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_below_0_1_path)
        plt.close()

    # adjacent_ratio_below_1_2 vs error
    if "adjacent_ratio_below_1_2" in before.columns and before["adjacent_ratio_below_1_2"].notna().any():
        ratio_below_1_2_path = output_dir / f"{file_prefix}_adjacent_ratio_-2.5_-1.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_below_1_2",
            feature_label="Adjacent Ratio (-2.5 to -1.5 Th)",
            output_path=ratio_below_1_2_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_below_1_2_path)
        plt.close()

    # adjacent_ratio_below_2_3 vs error
    if "adjacent_ratio_below_2_3" in before.columns and before["adjacent_ratio_below_2_3"].notna().any():
        ratio_below_2_3_path = output_dir / f"{file_prefix}_adjacent_ratio_-3.5_-2.5_vs_error.png"
        plot_adjacent_ion_feature_vs_error(
            before,
            after,
            feature_col="adjacent_ratio_below_2_3",
            feature_label="Adjacent Ratio (-3.5 to -2.5 Th)",
            output_path=ratio_below_2_3_path,
            use_log_scale=True,
            use_ppm=use_ppm,
            ylim=ylim,
        )
        generated_files.append(ratio_below_2_3_path)
        plt.close()

    # Feature importance
    if calibrator is not None and calibrator.training_stats:
        importance_path = output_dir / f"{file_prefix}_feature_importance.png"
        plot_feature_importance(calibrator, importance_path)
        generated_files.append(importance_path)
        plt.close()

    # Summary text file
    summary_path = output_dir / f"{file_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Mars Calibration QC Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Before Calibration:\n")
        f.write(f"  Matches: {len(before):,}\n")
        f.write(f"  Mean delta m/z: {before['delta_mz'].mean():.4f} Th\n")
        f.write(f"  Std delta m/z:  {before['delta_mz'].std():.4f} Th\n")
        f.write(f"  Median delta m/z: {before['delta_mz'].median():.4f} Th\n\n")

        if after is not None and "delta_mz_calibrated" in after.columns:
            f.write("After Calibration:\n")
            f.write(f"  Mean delta m/z: {after['delta_mz_calibrated'].mean():.4f} Th\n")
            f.write(f"  Std delta m/z:  {after['delta_mz_calibrated'].std():.4f} Th\n")
            f.write(f"  Median delta m/z: {after['delta_mz_calibrated'].median():.4f} Th\n\n")

            std_before = before["delta_mz"].std()
            std_after = after["delta_mz_calibrated"].std()
            improvement = (1 - std_after / std_before) * 100
            f.write(f"Improvement: {improvement:.1f}% reduction in std dev\n\n")

        if calibrator is not None:
            f.write("\n" + calibrator.get_stats_summary())

    generated_files.append(summary_path)
    logger.info(f"Generated QC report with {len(generated_files)} files in {output_dir}")

    return generated_files
