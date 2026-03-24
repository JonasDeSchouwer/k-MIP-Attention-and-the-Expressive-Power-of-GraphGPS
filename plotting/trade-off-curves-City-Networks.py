# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# %%
@dataclass
class MethodResult:
    """Container for a single method's performance results."""
    name: str
    training_time: float  # hours
    test_accuracy_mean: float
    test_accuracy_std: float
    color: Optional[str] = None
    marker: Optional[str] = None
    
    @property
    def training_time_minutes(self) -> float:
        """Convert training time from hours to minutes."""
        return self.training_time * 60

@dataclass 
class DatasetResults:
    """Container for all method results on a specific dataset."""
    name: str
    nodes: int
    edges: int
    methods: List[MethodResult]
    
    def get_method_by_name(self, name: str) -> Optional[MethodResult]:
        """Get a specific method result by name."""
        for method in self.methods:
            if method.name == name:
                return method
        return None
    
    def get_pareto_frontier(self) -> List[MethodResult]:
        """Get methods on the Pareto frontier (lower training time, higher accuracy)."""
        # Sort by training time
        sorted_methods = sorted(self.methods, key=lambda m: m.training_time)
        pareto_methods = []
        
        best_accuracy = -1
        for method in sorted_methods:
            if method.test_accuracy_mean > best_accuracy:
                pareto_methods.append(method)
                best_accuracy = method.test_accuracy_mean
                
        return pareto_methods


# %%
# Define color scheme and markers for different methods
METHOD_STYLING = {
    'GINE': {'color': '#1f77b4', 'marker': 'o'},
    'GCN': {'color': '#ff7f0e', 'marker': 's'},
    'GAT': {'color': '#2ca02c', 'marker': '^'},
    'GatedGCN': {'color': '#d62728', 'marker': 'v'},
    'Exphormer': {'color': '#9467bd', 'marker': 'D'},
    'GPS + Performer': {'color': '#8c564b', 'marker': 'p'},
    'GPS + k-MIP': {'color': '#e377c2', 'marker': 'X'},
    'GPS + BigBird': {'color': '#7f7f7f', 'marker': 'd'},
    'GPS + Transformer': {'color': '#bcbd22', 'marker': 'P'},
}

def apply_method_styling(method: MethodResult) -> MethodResult:
    """Apply consistent color and marker styling to a method."""
    styling = METHOD_STYLING.get(method.name, {'color': 'black', 'marker': 'o'})
    method.color = styling['color']
    method.marker = styling['marker']
    return method


# %%
# Paris Dataset Results (114k nodes, 183k edges)
paris_methods = [
    MethodResult("GINE", 0.048, 53.36, 0.23),
    MethodResult("GCN", 0.051, 52.93, 0.06),
    MethodResult("GAT", 0.071, 55.83, 0.42),
    MethodResult("GatedGCN", 0.092, 53.27, 0.10),
    MethodResult("Exphormer", 1.54, 51.40, 0.08),
    MethodResult("GPS + Performer", 2.62, 54.06, 0.27),
    MethodResult("GPS + k-MIP", 3.09, 53.62, 0.22),
    MethodResult("GPS + BigBird", 11.50, 53.53, 0.37),
]

# Apply styling to each method
paris_methods = [apply_method_styling(method) for method in paris_methods]

paris_dataset = DatasetResults(
    name="Paris",
    nodes=114000,
    edges=183000,
    methods=paris_methods
)


# %%
# Shanghai Dataset Results (184k nodes, 263k edges)
shanghai_methods = [
    MethodResult("GINE", 0.065, 63.35, 0.20),
    MethodResult("GCN", 0.066, 57.75, 0.24),
    MethodResult("GAT", 0.093, 72.53, 0.23),
    MethodResult("GatedGCN", 0.113, 68.80, 0.21),
    MethodResult("Exphormer", 2.16, 62.33, 0.16),
    MethodResult("GPS + Performer", 5.08, 67.27, 0.17),
    MethodResult("GPS + k-MIP", 6.45, 66.94, 0.44),
    MethodResult("GPS + BigBird", 19.21, 65.24, 0.17),
]

shanghai_methods = [apply_method_styling(method) for method in shanghai_methods]

shanghai_dataset = DatasetResults(
    name="Shanghai",
    nodes=184000,
    edges=263000,
    methods=shanghai_methods
)


# %%
# LA Dataset Results (241k nodes, 343k edges)
la_methods = [
    MethodResult("GINE", 0.087, 58.21, 0.56),
    MethodResult("GCN", 0.089, 56.65, 0.04),
    MethodResult("GAT", 0.125, 65.53, 0.65),
    MethodResult("GatedGCN", 0.150, 63.42, 0.12),
    MethodResult("Exphormer", 2.75, 58.15, 0.13),
    MethodResult("GPS + Performer", 7.87, 61.64, 0.24),
    MethodResult("GPS + k-MIP", 10.20, 61.72, 0.35),
    # GPS + BigBird omitted due to long training time
]

la_methods = [apply_method_styling(method) for method in la_methods]

la_dataset = DatasetResults(
    name="LA",
    nodes=241000,
    edges=343000,
    methods=la_methods
)


# %%
# London Dataset Results
london_methods = [
    MethodResult("GINE", 0.183, 57.60, 0.20),
    MethodResult("GCN", 0.193, 55.25, 0.06),
    MethodResult("GAT", 0.278, 57.19, 1.09),
    MethodResult("GatedGCN", 0.332, 61.47, 0.14),
    # Exphormer omited due to OOM
    # GPS + Transformer omitted due to OOM
    MethodResult("GPS + Performer", 13.66, 53.20, 0.00),
    MethodResult("GPS + k-MIP", 31.91, 56.05, 0.00),
    # GPS + BigBird omitted due to OOM
]

london_methods = [apply_method_styling(method) for method in london_methods]

london_dataset = DatasetResults(
    name="London",
    nodes=569000,  # Similar scale to LA
    edges=759000,
    methods=london_methods
)


# %%
# ShapeNet-Part Dataset Results (run on single V100 16GB)
shapenet_methods = [
    MethodResult("GINE", 3.27, 64.57, 0.35),
    MethodResult("GCN", 3.07, 60.18, 0.04),
    MethodResult("GAT", 4.08, 63.01, 0.17),
    MethodResult("GatedGCN", 7.50, 76.20, 0.32),
    MethodResult("Exphormer", 13.19, 82.62, 0.31),  # Run on A100 40GB
    MethodResult("GPS + Performer", 14.24, 77.36, 1.23),
    MethodResult("GPS + k-MIP", 14.84, 82.68, 0.64),
    MethodResult("GPS + BigBird", 27.35, 79.65, 0.98),
]

shapenet_methods = [apply_method_styling(method) for method in shapenet_methods]

shapenet_dataset = DatasetResults(
    name="ShapeNet-Part",
    nodes=0,  # Point cloud data
    edges=0,
    methods=shapenet_methods
)


# %%
# S3DIS Dataset Results (run on single A100 40GB)
s3dis_methods = [
    MethodResult("GINE", 30.58, 44.16, 0.62),
    MethodResult("GCN", 31.68, 39.98, 1.47),
    MethodResult("GAT", 33.61, 44.24, 1.14),
    MethodResult("GPS + k-MIP", 60.33, 67.99, 1.51),
    MethodResult("GPS + Performer", 62.32, 60.83, 0.56),
    MethodResult("GatedGCN", 63.40, 63.71, 1.28),
    MethodResult("Exphormer", 86.17, 68.37, 0.23),
    MethodResult("GPS + BigBird", 118.81, 67.92, 0.91),
]

s3dis_methods = [apply_method_styling(method) for method in s3dis_methods]

s3dis_dataset = DatasetResults(
    name="S3DIS",
    nodes=0,  # Point cloud data
    edges=0,
    methods=s3dis_methods
)


# %%
# Collect all datasets
all_datasets = [paris_dataset, shanghai_dataset, la_dataset, london_dataset, shapenet_dataset, s3dis_dataset]
city_datasets_with_london = [paris_dataset, shanghai_dataset, la_dataset, london_dataset]

# Print summary
print("Dataset Summary:")
print("=" * 50)
for dataset in all_datasets:
    print(f"{dataset.name}: {len(dataset.methods)} methods")
    if dataset.nodes > 0:
        print(f"  Nodes: {dataset.nodes:,}, Edges: {dataset.edges:,}")
    print(f"  Accuracy range: {min(m.test_accuracy_mean for m in dataset.methods):.1f} - {max(m.test_accuracy_mean for m in dataset.methods):.1f}")
    print(f"  Training time range: {min(m.training_time for m in dataset.methods):.3f} - {max(m.training_time for m in dataset.methods):.1f} hours")
    print()

# %%
def plot_combined_trade_offs(datasets: List[DatasetResults],
                             subplot_size: Tuple[float, float] = (3.5, 3.5),
                             output_dir: str = "plotting/plots",
                             log_scale_x: bool = True,
                             markersize: int = 12) -> plt.Figure:
    """
    Plot trade-off curves for multiple datasets in subplots.
    
    Args:
        datasets: List of dataset results to plot
        subplot_size: Size of each individual subplot (width, height) in inches
        output_dir: Directory to save plots
        log_scale_x: Whether to use log scale for x-axis
        markersize: Size of markers
    
    Returns:
        matplotlib Figure object
    """
    n_datasets = len(datasets)
    cols = 4
    rows = (n_datasets + cols - 1) // cols  # Ceiling division
    
    # Calculate figure size based on subplot size
    # Add padding for labels, titles, and spacing
    subplot_width, subplot_height = subplot_size
    
    # Extra width for y-label on leftmost plots (only Paris gets y-label)
    ylabel_width = 0.8  # inches for y-label
    spacing_width = 0.5  # inches between subplots
    margin_width = 1.0   # inches for margins
    
    # Extra height for titles and spacing
    title_height = 0.6   # inches for titles
    spacing_height = 0.4 # inches between subplot rows
    margin_height = 0.8  # inches for margins (reduced since no main title)
    
    # Calculate total figure size
    figwidth = (cols * subplot_width + 
                ylabel_width +  # Space for y-label on leftmost plot
                (cols - 1) * spacing_width + 
                margin_width)
    
    figheight = (rows * subplot_height + 
                 rows * title_height +
                 (rows - 1) * spacing_height + 
                 margin_height)
    
    fig, axes = plt.subplots(rows, cols, figsize=(figwidth, figheight))
    
    # Handle different axes shapes - always flatten to make indexing consistent
    if n_datasets == 1:
        axes = [axes]  # Single subplot
    else:
        axes = axes.flatten()  # Always flatten for consistent indexing
    
    # Plot each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Plot methods
        for method in dataset.methods:
            ax.errorbar(method.training_time, method.test_accuracy_mean,
                       yerr=method.test_accuracy_std,
                       fmt=method.marker, color=method.color,
                       label=method.name, markersize=markersize, capsize=3,
                       elinewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
        
        # Formatting
        ax.set_xlabel('Training Time (hours)')
        if dataset.name == "Paris":
            ax.set_ylabel('Test Accuracy (%)')
        
        title = f'$\\mathbf{{{dataset.name}}}$'
        if dataset.nodes > 0:
            title += f'\n{dataset.nodes//1000}k nodes'
        ax.set_title(title)
        
        if log_scale_x:
            ax.set_xscale('log')
        
        # Set custom grid intervals after plotting data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if not log_scale_x:
            # For linear scale, set x-ticks at intervals of 5
            x_min, x_max = xlim
            x_ticks = [i for i in range(0, int(x_max) + 5, 5) if i >= x_min]
            ax.set_xticks(x_ticks)
        
        # Set y-ticks at intervals of 2
        y_min, y_max = ylim
        y_ticks = [i for i in range(0, int(y_max) + 2, 2) if i >= y_min]
        ax.set_yticks(y_ticks)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_datasets, rows * cols):
        axes[i].set_visible(False)
    
    # Add legend to the figure in a separate box
    if n_datasets > 0:
        # Get handles and labels from the first subplot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='upper center', 
                  ncol=len(handles) // 2, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_tradeoff.pdf"), dpi=300, bbox_inches='tight')
    return fig


# %%
def plot_individual_city_curves(datasets: List[DatasetResults], 
                                output_dir: str = "plotting/plots",
                                figsize: Tuple[int, int] = (3, 3),
                                markersize: int = 10,
                                log_scale_x: bool = True) -> None:
    """
    Create individual trade-off curves for each city dataset with independent y-axes.
    
    Args:
        datasets: List of city dataset results to plot
        output_dir: Directory to save plots
        figsize: Figure size for each individual plot
        markersize: Size of markers
        log_scale_x: Whether to use log scale for x-axis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual plots
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each method
        for method in dataset.methods:
            ax.errorbar(method.training_time, method.test_accuracy_mean,
                       yerr=method.test_accuracy_std,
                       fmt=method.marker, color=method.color,
                       markersize=markersize, capsize=4,
                       elinewidth=2, markeredgewidth=1, markeredgecolor='white')
        
        # Formatting
        ax.set_xlabel('Training Time (hours)', fontsize=12)
        if dataset.name == "Paris":
            ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        # ax.set_title(f'{dataset.name}', fontsize=14, fontweight='bold')

        if log_scale_x:
            ax.set_xscale('log')
        
        # Set custom grid intervals after plotting data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if not log_scale_x:
            # For linear scale, set x-ticks at intervals of 5
            x_min, x_max = xlim
            x_ticks = [i for i in range(0, int(x_max) + 5, 5) if i >= x_min]
            ax.set_xticks(x_ticks)
        
        # Set y-ticks at intervals of 2
        y_min, y_max = ylim
        y_ticks = [i for i in range(0, int(y_max) + 2, 2) if i >= y_min]
        ax.set_yticks(y_ticks)
        
        ax.grid(True, alpha=0.3)
        
        # Remove legend from individual plots
        # ax.legend() - commented out
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"{dataset.name.lower()}_tradeoff.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        plt.close(fig)


def create_shared_legend(datasets: List[DatasetResults],
                        output_dir: str = "plotting/plots", 
                        figsize: Tuple[int, int] = (12, 1.5),
                        markersize: int = 10) -> None:
    """
    Create a shared horizontal legend for all methods.
    
    Args:
        datasets: List of datasets to extract methods from
        output_dir: Directory to save the legend
        figsize: Figure size for the legend
        markersize: Size of markers in legend
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all unique methods
    all_methods = {}
    for dataset in datasets:
        for method in dataset.methods:
            if method.name not in all_methods:
                all_methods[method.name] = method
    
    # Create legend figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dummy plots to generate legend handles
    handles = []
    labels = []
    for method_name, method in all_methods.items():
        handle = ax.scatter([], [], marker=method.marker, color=method.color,
                          s=markersize**2, edgecolors='white', linewidth=1,
                          label=method.name)
        handles.append(handle)
        labels.append(method.name)
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create horizontal legend
    legend = ax.legend(handles, labels, 
                      loc='center', 
                      ncol=len(labels), 
                      frameon=True,
                      fontsize=11,
                      markerscale=1.2,
                      columnspacing=1.5)
    
    # Save the legend
    filepath = os.path.join(output_dir, "shared_legend.pdf")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.close(fig)


# %%
# Analysis: Find best methods for each dataset
print("Best performing methods by dataset:")
print("=" * 60)

for dataset in all_datasets:
    print(f"\n{dataset.name} Dataset:")
    
    # Sort by accuracy score
    sorted_by_accuracy = sorted(dataset.methods, key=lambda m: m.test_accuracy_mean, reverse=True)
    print(f"  Highest accuracy: {sorted_by_accuracy[0].name} ({sorted_by_accuracy[0].test_accuracy_mean:.1f}±{sorted_by_accuracy[0].test_accuracy_std:.1f}%)")
    
    # Sort by training time
    sorted_by_time = sorted(dataset.methods, key=lambda m: m.training_time)
    print(f"  Fastest training: {sorted_by_time[0].name} ({sorted_by_time[0].training_time:.3f}h)")
    
    # Best efficiency (accuracy/time ratio)
    efficiency_scores = [(m.test_accuracy_mean / m.training_time, m) for m in dataset.methods]
    best_efficiency = max(efficiency_scores, key=lambda x: x[0])
    print(f"  Best efficiency: {best_efficiency[1].name} ({best_efficiency[0]:.1f} accuracy/hour)")
    
    # Pareto frontier
    pareto = dataset.get_pareto_frontier()
    pareto_names = [m.name for m in pareto]
    print(f"  Pareto frontier: {', '.join(pareto_names)}")


# %%
# Generate individual city plots and shared legend for LaTeX subfigures
# print("\nGenerating individual city plots with independent y-axes...")
# plot_individual_city_curves(city_datasets_with_london,
#                            output_dir="figures",
#                            figsize=(5, 5),
#                            markersize=12)

print("\nGenerating shared horizontal legend...")
create_shared_legend(city_datasets_with_london,
                    output_dir="figures",
                    figsize=(14, 1.5),
                    markersize=12)

print("\nGenerating combined plot...")
plot_combined_trade_offs(city_datasets_with_london,
                         subplot_size=(2, 2),
                         output_dir="figures",
                         log_scale_x=False,
                         markersize=10)

