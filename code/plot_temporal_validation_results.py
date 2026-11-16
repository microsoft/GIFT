"""
Plot temporal validation results from multi-seed evaluation
- Separate significant and non-significant metrics
- Avoid mixing metrics with different magnitudes
- Clean, publication-ready visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
RESULTS_PATH = "validation_results_chembl36/soft_edge_validation_multi_seed.json"
OUTPUT_DIR = "validation_results_chembl36"

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_results():
    """Load temporal validation results"""
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)

def extract_metric_data(results, metric_key):
    """Extract baseline and soft-edge values for a metric across seeds"""
    baseline_vals = []
    softedge_vals = []
    
    for seed_data in results['per_seed_results'].values():
        baseline_vals.append(seed_data['baseline'][metric_key])
        softedge_vals.append(seed_data['soft_edge'][metric_key])
    
    return np.array(baseline_vals), np.array(softedge_vals)

def create_temporal_validation_plots(results):
    """Create improved temporal validation plots"""
    
    # Extract metrics and check which are significant
    stats = results['aggregate_statistics']
    
    # Group 1: Significant metrics with similar scale (pChEMBL values ~5-7)
    significant_similar_scale = []
    if stats['avg_top10_pchembl']['is_significant']:
        significant_similar_scale.append(('avg_top10_pchembl', 'Avg pChEMBL\nTop-10'))
    if stats['enrichment']['is_significant']:
        significant_similar_scale.append(('enrichment', 'Enrichment\n(Top-Bottom)'))

    if not significant_similar_scale:
        significant_similar_scale = [
            ('avg_top10_pchembl', 'Avg pChEMBL\nTop-10'),
            ('enrichment', 'Enrichment\n(Top-Bottom)'),
        ]
    
    # Group 2: Other metrics (different scales)
    other_metrics = []
    if stats['n_strong_in_top10']['is_significant']:
        other_metrics.append(('n_strong_in_top10', 'Strong (≥7.0)\nin Top-10'))
    if not stats['avg_percentile_strong']['is_significant']:
        other_metrics.append(('avg_percentile_strong', 'Avg Percentile\nof Strong'))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.3)
    
    # Colors
    color_baseline = '#7570b3'  # Purple
    color_softedge = '#d95f02'  # Orange
    
    # ========================================================================
    # PLOT 1: ONLY Significant metrics with similar magnitudes
    # ========================================================================
    ax1 = axes[0]
    
    if len(significant_similar_scale) > 0:
        x_positions = np.arange(len(significant_similar_scale))
        width = 0.35
        
        baseline_means = []
        baseline_stds = []
        softedge_means = []
        softedge_stds = []
        labels = []
        
        for metric_key, label in significant_similar_scale:
            baseline_vals, softedge_vals = extract_metric_data(results, metric_key)
            
            baseline_means.append(np.mean(baseline_vals))
            baseline_stds.append(np.std(baseline_vals))
            softedge_means.append(np.mean(softedge_vals))
            softedge_stds.append(np.std(softedge_vals))
            labels.append(label)
        
        # Plot bars
        ax1.bar(x_positions - width/2, baseline_means, width,
                yerr=baseline_stds, capsize=5, alpha=0.8,
                color=color_baseline, label='Baseline', edgecolor='black', linewidth=1.2)
        
        ax1.bar(x_positions + width/2, softedge_means, width,
                yerr=softedge_stds, capsize=5, alpha=0.8,
                color=color_softedge, label='Soft-Edge', edgecolor='black', linewidth=1.2)
        
        # Add significance markers
        for i, (metric_key, _) in enumerate(significant_similar_scale):
            if stats[metric_key]['is_significant']:
                y_max = max(baseline_means[i] + baseline_stds[i],
                           softedge_means[i] + softedge_stds[i])
                ax1.text(i, y_max + 0.15, '**', ha='center', fontsize=14, fontweight='bold')
        
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('(A) Performance Comparison Across Metrics', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'No significant metrics\nwith similar scale', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('(A) Performance Comparison Across Metrics', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 2: Per-seed improvement in Top-10 ranking
    # ========================================================================
    ax2 = axes[1]
    
    # Calculate per-seed improvement
    seeds = sorted([int(s) for s in results['per_seed_results'].keys()])
    improvements = []
    
    for seed in seeds:
        seed_str = str(seed)
        baseline_val = results['per_seed_results'][seed_str]['baseline']['avg_top10_pchembl']
        softedge_val = results['per_seed_results'][seed_str]['soft_edge']['avg_top10_pchembl']
        improvement = softedge_val - baseline_val
        improvements.append(improvement)
    
    improvements = np.array(improvements)
    
    # Count positive improvements
    n_positive = np.sum(improvements > 0)
    pct_positive = n_positive / len(improvements) * 100
    
    # Plot scatter
    ax2.scatter(seeds, improvements, alpha=0.7, s=60, color=color_softedge, edgecolors='black', linewidth=1)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(np.mean(improvements), color=color_baseline, linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(improvements):.3f}')
    
    # Highlight positive/negative
    for i, (seed, imp) in enumerate(zip(seeds, improvements)):
        if imp < 0:
            ax2.scatter(seed, imp, s=80, color='red', alpha=0.8, marker='o', edgecolors='darkred', linewidth=1.5)
    
    ax2.set_xlabel('Random Seed', fontsize=11)
    ax2.set_ylabel('Δ Avg pChEMBL Top-10\n(Soft-Edge − Baseline)', fontsize=11)
    ax2.set_title(f'(B) Per-Seed Improvement in Top-10 Ranking', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation
    ax2.text(0.05, 0.95, f'Seeds with improvement: {n_positive}/{len(seeds)} ({pct_positive:.1f}%)',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    n_seeds = results['num_seeds']
    
    # Add note about significance / saved result mode
    if results.get('num_seeds', 0) > 1:
        note_text = '** indicates p < 0.05 (paired t-test)'
    else:
        note_text = 'Single-seed strict saved result; significance not computed'

    fig.text(0.5, 0.02, note_text,
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = Path(OUTPUT_DIR) / 'temporal_validation_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("PLOTTING TEMPORAL VALIDATION RESULTS")
    print("="*80)
    
    # Load results
    print(f"\nLoading results from: {RESULTS_PATH}")
    results = load_results()
    
    print(f"Found results for {results['num_seeds']} seeds")
    print(f"Evaluated on {results['num_new_interactions']} new interactions")
    
    # Create plots
    print("\nCreating visualization...")
    create_temporal_validation_plots(results)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
