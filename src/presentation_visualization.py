"""
Presentation-ready visualizations for resource availability models.

Provides both micro (individual resource) and macro (aggregated) views
of availability patterns using heatmaps, bar charts, and comparisons.

Uses REAL data from the BPI 2017 CSV via ResourceAvailabilityModel.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the real resource availability model and dataclasses
from resource_availability_1_5 import (
    ResourceAvailabilityModel,
    AvailabilityArtifactBasic,
    AvailabilityArtifactAdvanced
)

def load_real_artifacts(csv_path: str, tau: float = 0.50, tau_month: float = 0.50):
    """
    Load real artifacts from BPI CSV using ResourceAvailabilityModel.
    
    Args:
        csv_path: Path to the BPI 2017 CSV file
        tau: Threshold for basic model
        tau_month: Threshold for advanced model
        
    Returns:
        Tuple of (artifact_basic, artifact_advanced)
    """
    print(f"[Visualization] Loading basic model from {csv_path}...")
    basic_model = ResourceAvailabilityModel(
        csv_path=csv_path,
        mode="basic",
        tau=tau
    )
    
    print(f"[Visualization] Loading advanced model from {csv_path}...")
    advanced_model = ResourceAvailabilityModel(
        csv_path=csv_path,
        mode="advanced",
        tau_month=tau_month
    )
    
    return basic_model.artifact, advanced_model.artifact

def get_weekday_names() -> List[str]:
    """Return list of weekday names (Monday-Sunday)."""
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_month_names() -> List[str]:
    """Return list of month names (1-indexed, January-December)."""
    return ["", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"]


def get_2hour_bin_labels() -> List[str]:
    """Return list of 2-hour time bin labels."""
    return ["00-02", "02-04", "04-06", "06-08", "08-10", "10-12",
            "12-14", "14-16", "16-18", "18-20", "20-22", "22-24"]


def get_week_type_names() -> List[str]:
    """Return week type names."""
    return ["Even Weeks", "Odd Weeks"]


# ============================================================================
# Visualization 1 & 2: Basic Model - Aggregated and Resource-Specific
# ============================================================================

def plot_basic_aggregated_heatmaps(artifact_basic: AvailabilityArtifactBasic,
                                   output_dir: str = "outputs") -> None:
    """
    Create side-by-side heatmaps showing aggregated resource availability
    for even and odd weeks across a full week and 24-hour period.
    
    Macro view: Shows how many resources are available at each week_type/weekday/hour.
    
    Args:
        artifact_basic: AvailabilityArtifactBasic instance
        output_dir: Directory to save the plot
    """
    weekday_names = get_weekday_names()
    hours = list(range(24))
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Basic Model: Aggregated Resource Availability by Week Type",
                 fontsize=16, fontweight='bold', y=1.02)
    
    # First pass: collect all data to compute global min/max
    all_data = []
    for week_type in range(2):
        data = [[0.0 for _ in range(24)] for _ in range(7)]
        for weekday in range(7):
            for hour in range(24):
                key = (week_type, weekday, hour)
                data[weekday][hour] = len(artifact_basic.avail_index.get(key, []))
        all_data.append(data)
    
    # Compute global vmin and vmax
    flat_values = [val for week_data in all_data for row in week_data for val in row]
    vmin = min(flat_values) if flat_values else 0
    vmax = max(flat_values) if flat_values else 1
    
    # Second pass: plot with normalized scale
    for week_type in range(2):
        df = pd.DataFrame(all_data[week_type], index=weekday_names, columns=hours)
        
        # Plot heatmap
        ax = axes[week_type]
        sns.heatmap(df, annot=False, cmap="YlOrRd", ax=ax,
                    cbar_kws={'label': 'Available Resources'},
                    vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(get_week_type_names()[week_type], fontsize=12, fontweight='bold')
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Weekday", fontsize=11)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basic_aggregated_heatmaps.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_basic_resource_heatmaps(artifact_basic: AvailabilityArtifactBasic,
                                 resource_id: str,
                                 output_dir: str = "outputs") -> None:
    """
    Create side-by-side heatmaps showing availability pattern of a specific resource
    for even and odd weeks across a full week and 24-hour period.
    
    Micro view: Shows where a specific resource is available (1) or not (0).
    
    Args:
        artifact_basic: AvailabilityArtifactBasic instance
        resource_id: ID of the resource to visualize
        output_dir: Directory to save the plot
    """
    weekday_names = get_weekday_names()
    hours = list(range(24))
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Basic Model: Availability Pattern for Resource '{resource_id}'",
                 fontsize=16, fontweight='bold', y=1.02)
    
    # First pass: collect data for both week types
    all_data = []
    for week_type in range(2):
        data = [[0.0 for _ in range(24)] for _ in range(7)]
        for weekday in range(7):
            for hour in range(24):
                key = (week_type, weekday, hour)
                resources = artifact_basic.avail_index.get(key, [])
                data[weekday][hour] = 1 if resource_id in resources else 0
        all_data.append(data)
    
    # For binary data, vmin=0 and vmax=1
    vmin, vmax = 0, 1
    
    # Plot both week types
    for week_type in range(2):
        df = pd.DataFrame(all_data[week_type], index=weekday_names, columns=hours)
        
        # Plot heatmap
        ax = axes[week_type]
        sns.heatmap(df, annot=False, cmap="RdYlGn", ax=ax,
                    cbar=False, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(get_week_type_names()[week_type], fontsize=12, fontweight='bold')
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Weekday", fontsize=11)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basic_resource_heatmaps.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Visualization 3 & 4: Advanced Model - Aggregated and Resource-Specific
# ============================================================================

def plot_advanced_aggregated_month_heatmaps(artifact_advanced: AvailabilityArtifactAdvanced,
                                            months: List[int] = None,
                                            output_dir: str = "outputs") -> None:
    """
    Create heatmaps showing aggregated resource availability by month,
    weekday, and 2-hour time buckets.
    
    Macro view: Shows how many resources are available at each month/weekday/bucket2h.
    
    Args:
        artifact_advanced: AvailabilityArtifactAdvanced instance
        months: List of month numbers (1-12). Defaults to [1, 6, 12]
        output_dir: Directory to save the plot
    """
    if months is None:
        months = [1, 6, 12]
    
    weekday_names = get_weekday_names()
    bin_labels = get_2hour_bin_labels()
    month_names = get_month_names()
    
    # Create figure with subplots (one per month)
    num_months = len(months)
    fig, axes = plt.subplots(num_months, 1, figsize=(14, 4 * num_months))
    
    # Handle single subplot case
    if num_months == 1:
        axes = [axes]
    
    fig.suptitle("Advanced Model: Aggregated Resource Availability by Month",
                 fontsize=16, fontweight='bold', y=0.995)
    
    # First pass: collect all month data to compute global min/max
    all_month_data = []
    for month in months:
        data = [[0.0 for _ in range(12)] for _ in range(7)]
        for weekday in range(7):
            for bucket2h in range(12):
                key = (month, weekday, bucket2h)
                data[weekday][bucket2h] = len(artifact_advanced.monthly_index.get(key, []))
        all_month_data.append(data)
    
    # Compute global vmin and vmax
    flat_values = [val for month_data in all_month_data for row in month_data for val in row]
    vmin = min(flat_values) if flat_values else 0
    vmax = max(flat_values) if flat_values else 1
    
    # Second pass: plot with normalized scale
    for idx, month in enumerate(months):
        df = pd.DataFrame(all_month_data[idx], index=weekday_names, columns=bin_labels)
        
        # Plot heatmap
        ax = axes[idx]
        sns.heatmap(df, annot=False, cmap="YlOrRd", ax=ax,
                    cbar_kws={'label': 'Available Resources'},
                    vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(f"{month_names[month]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("2-Hour Time Bucket", fontsize=11)
        ax.set_ylabel("Weekday", fontsize=11)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "advanced_aggregated_heatmaps.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_advanced_resource_month_heatmaps(artifact_advanced: AvailabilityArtifactAdvanced,
                                          resource_id: str,
                                          months: List[int] = None,
                                          output_dir: str = "outputs") -> None:
    """
    Create heatmaps showing availability pattern of a specific resource by month,
    weekday, and 2-hour time buckets.
    
    Micro view: Shows where a specific resource is available (1) or not (0).
    
    Args:
        artifact_advanced: AvailabilityArtifactAdvanced instance
        resource_id: ID of the resource to visualize
        months: List of month numbers (1-12). Defaults to [1, 6, 12]
        output_dir: Directory to save the plot
    """
    if months is None:
        months = [1, 6, 12]
    
    weekday_names = get_weekday_names()
    bin_labels = get_2hour_bin_labels()
    month_names = get_month_names()
    
    # Create figure with subplots (one per month)
    num_months = len(months)
    fig, axes = plt.subplots(num_months, 1, figsize=(14, 4 * num_months))
    
    # Handle single subplot case
    if num_months == 1:
        axes = [axes]
    
    fig.suptitle(f"Advanced Model: Availability Pattern for Resource '{resource_id}'",
                 fontsize=16, fontweight='bold', y=0.995)
    
    # For binary data, vmin=0 and vmax=1
    vmin, vmax = 0, 1
    
    # Plot all months with same scale
    for idx, month in enumerate(months):
        data = [[0.0 for _ in range(12)] for _ in range(7)]
        for weekday in range(7):
            for bucket2h in range(12):
                key = (month, weekday, bucket2h)
                resources = artifact_advanced.monthly_index.get(key, [])
                data[weekday][bucket2h] = 1 if resource_id in resources else 0
        
        df = pd.DataFrame(data, index=weekday_names, columns=bin_labels)
        
        # Plot heatmap
        ax = axes[idx]
        sns.heatmap(df, annot=False, cmap="RdYlGn", ax=ax,
                    cbar=False, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(f"{month_names[month]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("2-Hour Time Bucket", fontsize=11)
        ax.set_ylabel("Weekday", fontsize=11)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "advanced_resource_heatmaps.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Visualization 5: Top Resources Bar Chart
# ============================================================================

def plot_top_resources_basic(artifact_basic: AvailabilityArtifactBasic,
                             top_n: int = 10,
                             output_dir: str = "outputs") -> None:
    """
    Create a horizontal bar chart showing the top N most available resources
    in the basic model (by number of slots they appear in).
    
    Args:
        artifact_basic: AvailabilityArtifactBasic instance
        top_n: Number of top resources to display
        output_dir: Directory to save the plot
    """
    # Count resource appearances across all slots
    resource_counts = Counter()
    for resources in artifact_basic.avail_index.values():
        resource_counts.update(resources)
    
    # Get top N resources
    top_resources = dict(resource_counts.most_common(top_n))
    
    # Create dataframe
    df = pd.DataFrame(list(top_resources.items()), columns=['Resource ID', 'Availability Slots'])
    df = df.sort_values('Availability Slots')
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df['Resource ID'], df['Availability Slots'], color='steelblue')
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel("Number of Availability Slots", fontsize=12, fontweight='bold')
    ax.set_ylabel("Resource ID", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {top_n} Most Available Resources (Basic Model)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "top_resources_basic.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Visualization 6: Basic vs Advanced Comparison
# ============================================================================

def compare_basic_vs_advanced(artifact_basic: AvailabilityArtifactBasic,
                              artifact_advanced: AvailabilityArtifactAdvanced,
                              output_dir: str = "outputs") -> None:
    """
    Create a bar chart comparing total resource appearances (resource-slot pairs)
    between the basic and advanced models.
    
    Args:
        artifact_basic: AvailabilityArtifactBasic instance
        artifact_advanced: AvailabilityArtifactAdvanced instance
        output_dir: Directory to save the plot
    """
    # Count total appearances in basic model
    basic_total = sum(len(resources) for resources in artifact_basic.avail_index.values())
    
    # Count total appearances in advanced model
    advanced_total = sum(len(resources) for resources in artifact_advanced.monthly_index.values())
    
    # Create comparison data
    models = ['Basic Model\n(Weekly)', 'Advanced Model\n(Monthly)']
    totals = [basic_total, advanced_total]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, totals, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{int(total)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel("Total Resource Availability Entries", fontsize=12, fontweight='bold')
    ax.set_title("Comparison: Basic vs Advanced Availability Model",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basic_vs_advanced_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

def main(csv_path: str, tau: float = 0.50, tau_month: float = 0.50):
    """
    Generate all visualizations using real BPI data.
    
    Args:
        csv_path: Path to the BPI 2017 CSV file (e.g., "data/bpi2017.csv")
        tau: Threshold for basic model
        tau_month: Threshold for advanced model
    """
    print("=" * 80)
    print("RESOURCE AVAILABILITY VISUALIZATION - REAL DATA")
    print("=" * 80)
    
    # Load real artifacts from BPI CSV
    artifact_basic, artifact_advanced = load_real_artifacts(csv_path, tau, tau_month)
    
    print(f"\n[Data Summary]")
    print(f"  Basic model: {len(artifact_basic.avail_index)} time slots")
    print(f"  Advanced model: {len(artifact_advanced.monthly_index)} time slots")
    
    # Get all unique resources to show data coverage
    all_basic_resources = set()
    for resources in artifact_basic.avail_index.values():
        all_basic_resources.update(resources)
    print(f"  Unique resources: {len(all_basic_resources)}")
    
    print("\nGenerating visualizations...\n")
    
    # 1. Basic aggregated heatmaps (macro view)
    print("1. Creating basic aggregated heatmaps...")
    plot_basic_aggregated_heatmaps(artifact_basic)
    
    # 2. Basic resource heatmaps (micro view) - use first available resource
    if all_basic_resources:
        resource_id = sorted(list(all_basic_resources))[0]
        print(f"2. Creating basic resource heatmaps for {resource_id}...")
        plot_basic_resource_heatmaps(artifact_basic, resource_id=resource_id)
    else:
        print("2. Skipping resource heatmaps (no resources found)")
    
    # 3. Advanced aggregated month heatmaps (macro view)
    print("3. Creating advanced aggregated month heatmaps...")
    plot_advanced_aggregated_month_heatmaps(artifact_advanced, months=[1, 6, 12])
    
    # 4. Advanced resource month heatmaps (micro view)
    if all_basic_resources:
        resource_id = sorted(list(all_basic_resources))[0]
        print(f"4. Creating advanced resource month heatmaps for {resource_id}...")
        plot_advanced_resource_month_heatmaps(artifact_advanced, resource_id=resource_id, months=[1, 6, 12])
    else:
        print("4. Skipping resource month heatmaps (no resources found)")
    
    # 5. Top resources bar chart
    print("5. Creating top resources bar chart...")
    plot_top_resources_basic(artifact_basic, top_n=10)
    
    # 6. Basic vs Advanced comparison
    print("6. Creating basic vs advanced comparison...")
    compare_basic_vs_advanced(artifact_basic, artifact_advanced)
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("Output directory: outputs/")
    print("=" * 80)


if __name__ == "__main__":
    # Real CSV path (adjust if needed)
    project_root = Path(__file__).resolve().parents[1]
    csv_path = str(project_root / "data" / "bpi2017.csv")
    
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        exit(1)
    
    main(csv_path=csv_path, tau=0.50, tau_month=0.50)
