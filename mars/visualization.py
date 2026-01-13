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
    xlim: tuple[float, float] = (-0.5, 0.5),
    intensity_weighted: bool = False,
) -> plt.Figure:
    """Plot histogram of delta m/z distribution, before and after calibration.

    Args:
        before: DataFrame with 'delta_mz', 'observed_intensity' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure (optional)
        title: Plot title
        bins: Number of histogram bins
        xlim: X-axis limits
        intensity_weighted: If True, use intensity-weighted histogram

    Returns:
        Matplotlib Figure
    """
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

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="0 Th")
        ax.axvline(
            stats["median"],
            color="orange",
            linestyle="-",
            linewidth=1.5,
            label=f"Median: {stats['median']:.4f} Th",
        )

        ax.set_xlabel("Delta m/z (Th)", fontsize=12)
        ax.set_title(subplot_title, fontsize=14)
        ax.legend(loc="upper right")
        ax.set_xlim(xlim)

        # Robust statistics text box
        stats_text = (
            f"n = {len(delta):,}\n"
            f"Median = {stats['median']:.4f}\n"
            f"MAD = {stats['mad']:.4f}\n"
            f"RMS = {stats['rms']:.4f}\n"
            f"Wt.Mean = {stats['wmean']:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        return stats

    # Before calibration
    stats_before = make_histogram(before, ax_before, "delta_mz", "Before Calibration", "steelblue")

    # After calibration
    stats_after = None
    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        stats_after = make_histogram(after, ax_after, delta_col, "After Calibration", "forestgreen")

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
    vmin: float = -0.25,
    vmax: float = 0.25,
) -> plt.Figure:
    """Plot 2D heatmap of delta m/z (X=RT, Y=fragment m/z, Color=median delta).

    Args:
        before: DataFrame with 'rt', 'fragment_mz', 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        rt_bins: Number of RT bins
        mz_bins: Number of m/z bins
        vmin: Color scale minimum
        vmax: Color scale maximum

    Returns:
        Matplotlib Figure
    """
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
            vmin=vmin,
            vmax=vmax,
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
        cbar.set_label("Median Delta m/z (Th)", fontsize=10)

        return heatmap_data

    # Before calibration
    make_heatmap(before, ax_before, "delta_mz", "Before Calibration")

    # After calibration
    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_heatmap(after, ax_after, delta_col, "After Calibration")

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

    features = list(importance.keys())
    values = list(importance.values())

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
    for i, (f, v) in enumerate(zip(features, values)):
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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of log10 intensity vs delta m/z.

    Args:
        before: DataFrame with 'observed_intensity' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        x = np.log10(np.clip(df["observed_intensity"], 1, None))
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="hot_r", mincnt=1)
        ax.axhline(0, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Intensity)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of retention time vs delta m/z.

    Args:
        before: DataFrame with 'rt' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="hot_r", mincnt=1)
        ax.axhline(0, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(
            "Acquisition Time (s)" if time_col == "absolute_time" else "Retention Time (min)",
            fontsize=12,
        )
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of fragment m/z vs delta m/z.

    Args:
        before: DataFrame with 'fragment_mz' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
    if after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_before, ax_after = axes
    else:
        fig, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        ax_after = None

    def make_hexbin(df: pd.DataFrame, ax: plt.Axes, delta_col: str, subplot_title: str):
        x = df["fragment_mz"].values
        y = df[delta_col].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="hot_r", mincnt=1)
        ax.axhline(0, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Fragment m/z (Th)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of log10 spectrum TIC vs delta m/z.

    Args:
        before: DataFrame with 'tic' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="hot_r", mincnt=1)
        ax.axhline(0, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Spectrum TIC)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Fragment count")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of injection time vs delta m/z.

    Args:
        before: DataFrame with 'injection_time' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Log10(Fragment count)")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of log10 fragment ions vs delta m/z.

    Args:
        before: DataFrame with 'fragment_ions' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(Fragment Ions)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Log10(Fragment count)")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of TIC×injection_time vs delta m/z.

    Args:
        before: DataFrame with 'tic_injection_time' and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        output_path: Path to save figure
        title: Plot title
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Log10(TIC×Injection Time)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Log10(Fragment count)")

    make_hexbin(before, ax_before, "delta_mz", "Before Calibration")

    if ax_after is not None and after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, ax_after, delta_col, "After Calibration")

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
    ylim: tuple[float, float] = (-0.25, 0.25),
    gridsize: int = 100,
) -> plt.Figure:
    """Plot 2D hexbin of a single temperature feature vs delta m/z.

    Args:
        before: DataFrame with temperature column and 'delta_mz' columns
        after: Optional DataFrame with 'delta_mz_calibrated' column
        temp_col: Column name for temperature (e.g., 'rfa2_temp', 'rfc2_temp')
        temp_label: Human-readable label for the temperature
        output_path: Path to save figure
        title: Plot title (defaults to "{temp_label} vs Mass Error")
        ylim: Y-axis limits for mass error
        gridsize: Hexbin grid size

    Returns:
        Matplotlib Figure
    """
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

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(f"{temp_label} Temperature (°C)", fontsize=12)
        ax.set_ylabel("Delta m/z (Th)", fontsize=12)
        ax.set_ylim(ylim)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_title(subplot_title, fontsize=12)
        plt.colorbar(hb, ax=ax, label="Log10(Fragment count)")

    # Before calibration
    make_hexbin(before, axes[0, 0], "delta_mz", "Before Calibration")

    # After calibration
    if after is not None:
        delta_col = "delta_mz_calibrated" if "delta_mz_calibrated" in after.columns else "delta_mz"
        make_hexbin(after, axes[0, 1], delta_col, "After Calibration")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {temp_label} temperature vs error plot to {output_path}")

    return fig


def generate_qc_report(
    before: pd.DataFrame,
    after: pd.DataFrame | None,
    calibrator,
    output_dir: Path | str,
    file_prefix: str = "mars_qc",
) -> list[Path]:
    """Generate full QC report with all plots.

    Args:
        before: DataFrame with matches before calibration
        after: DataFrame with matches after calibration (with delta_mz_calibrated)
        calibrator: Trained MzCalibrator
        output_dir: Output directory for plots
        file_prefix: Prefix for output files

    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # Histogram
    hist_path = output_dir / f"{file_prefix}_histogram.png"
    plot_delta_mz_histogram(before, after, hist_path)
    generated_files.append(hist_path)
    plt.close()

    # Heatmap (RT x fragment m/z)
    heatmap_path = output_dir / f"{file_prefix}_heatmap.png"
    plot_delta_mz_heatmap(before, after, heatmap_path)
    generated_files.append(heatmap_path)
    plt.close()

    # Intensity vs error
    intensity_path = output_dir / f"{file_prefix}_intensity_vs_error.png"
    plot_intensity_vs_error(before, after, intensity_path)
    generated_files.append(intensity_path)
    plt.close()

    # RT vs error
    rt_path = output_dir / f"{file_prefix}_rt_vs_error.png"
    plot_rt_vs_error(before, after, rt_path)
    generated_files.append(rt_path)
    plt.close()

    # Fragment m/z vs error
    mz_path = output_dir / f"{file_prefix}_mz_vs_error.png"
    plot_fragment_mz_vs_error(before, after, mz_path)
    generated_files.append(mz_path)
    plt.close()

    # TIC vs error
    tic_path = output_dir / f"{file_prefix}_tic_vs_error.png"
    plot_tic_vs_error(before, after, tic_path)
    generated_files.append(tic_path)
    plt.close()

    # Injection time vs error
    injection_time_path = output_dir / f"{file_prefix}_injection_time_vs_error.png"
    plot_injection_time_vs_error(before, after, injection_time_path)
    generated_files.append(injection_time_path)
    plt.close()

    # TIC×Injection time vs error
    tic_injection_time_path = output_dir / f"{file_prefix}_tic_injection_time_vs_error.png"
    plot_tic_injection_time_vs_error(before, after, tic_injection_time_path)
    generated_files.append(tic_injection_time_path)
    plt.close()

    # Fragment ions vs error
    fragment_ions_path = output_dir / f"{file_prefix}_fragment_ions_vs_error.png"
    plot_fragment_ions_vs_error(before, after, fragment_ions_path)
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
        )
        generated_files.append(rfc2_path)
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
