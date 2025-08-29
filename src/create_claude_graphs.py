#!/usr/bin/env python3
"""
Script to create comprehensive graphs for Claude study results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the Claude results data."""
    data_path = Path('shared_data/runs_claude/20250727_212029/per_item_scores.csv')
    df = pd.read_csv(data_path)
    return df

def create_valence_comparison_plots(df):
    """Create plots comparing performance across valence types."""
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Claude Performance by Prompt Valence', fontsize=16, fontweight='bold')
    
    # 1. Total Score Distribution
    ax1 = axes[0, 0]
    valence_order = ['Neutral', 'Supportive', 'Threatening']
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    for i, valence in enumerate(valence_order):
        data = df[df['type'] == valence]['total_score']
        ax1.hist(data, alpha=0.7, label=valence, color=colors[i], bins=10, edgecolor='black')
    
    ax1.set_xlabel('Total Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Scores by Valence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean Scores by Category
    ax2 = axes[0, 1]
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality']
    
    means_by_valence = df.groupby('type')[categories].mean()
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, valence in enumerate(valence_order):
        values = means_by_valence.loc[valence].values
        ax2.bar(x + i*width, values, width, label=valence, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Evaluation Categories')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Mean Scores by Category and Valence')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box Plot of Total Scores
    ax3 = axes[1, 0]
    box_data = [df[df['type'] == valence]['total_score'] for valence in valence_order]
    bp = ax3.boxplot(box_data, tick_labels=valence_order, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Total Score')
    ax3.set_title('Total Score Distribution (Box Plot)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    summary_stats = df.groupby('type')['total_score'].agg(['mean', 'std', 'count']).round(2)
    
    # Create a table
    table_data = []
    for valence in valence_order:
        row = [valence, f"{summary_stats.loc[valence, 'mean']:.2f}", 
               f"{summary_stats.loc[valence, 'std']:.2f}", 
               int(summary_stats.loc[valence, 'count'])]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Valence', 'Mean', 'Std Dev', 'Count'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax4.set_title('Summary Statistics by Valence')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('shared_data/runs_claude/20250727_212029/valence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_topic_analysis_plots(df):
    """Create plots analyzing performance by topic."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Claude Performance by Historical Topic', fontsize=16, fontweight='bold')
    
    # 1. Top 10 Topics by Average Score
    ax1 = axes[0, 0]
    topic_means = df.groupby('topic')['total_score'].mean().sort_values(ascending=True)
    top_10 = topic_means.tail(10)
    
    bars = ax1.barh(range(len(top_10)), top_10.values, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10.index, fontsize=9)
    ax1.set_xlabel('Average Total Score')
    ax1.set_title('Top 10 Topics by Average Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                ha='left', va='center', fontsize=8)
    
    # 2. Topic Performance Heatmap
    ax2 = axes[0, 1]
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality']
    
    topic_category_means = df.groupby('topic')[categories].mean()
    
    # Create heatmap
    sns.heatmap(topic_category_means.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax2, cbar_kws={'label': 'Mean Score'})
    ax2.set_title('Topic Performance Heatmap')
    ax2.set_xlabel('Topic')
    ax2.set_ylabel('Evaluation Category')
    
    # 3. Valence Performance by Topic
    ax3 = axes[1, 0]
    topic_valence_means = df.groupby(['topic', 'type'])['total_score'].mean().unstack()
    
    topic_valence_means.plot(kind='bar', ax=ax3, color=['#2E8B57', '#4682B4', '#CD5C5C'], alpha=0.8)
    ax3.set_xlabel('Topic')
    ax3.set_ylabel('Average Total Score')
    ax3.set_title('Valence Performance by Topic')
    ax3.legend(title='Valence')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Score Distribution by Topic
    ax4 = axes[1, 1]
    # Select top 8 topics for clarity
    top_topics = df.groupby('topic')['total_score'].mean().nlargest(8).index
    top_topic_data = df[df['topic'].isin(top_topics)]
    
    sns.boxplot(data=top_topic_data, x='topic', y='total_score', ax=ax4)
    ax4.set_xlabel('Topic')
    ax4.set_ylabel('Total Score')
    ax4.set_title('Score Distribution by Top Topics')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shared_data/runs_claude/20250727_212029/topic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_category_plots(df):
    """Create detailed plots for each evaluation category."""
    
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Detailed Analysis by Evaluation Category', fontsize=16, fontweight='bold')
    
    valence_order = ['Neutral', 'Supportive', 'Threatening']
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    for i, category in enumerate(categories):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Create violin plot
        data_to_plot = [df[df['type'] == valence][category] for valence in valence_order]
        parts = ax.violinplot(data_to_plot, positions=range(len(valence_order)))
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(valence_order)))
        ax.set_xticklabels(valence_order, rotation=45)
        ax.set_ylabel('Score')
        ax.set_title(category.replace('_', '\n'))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)
    
    # Remove the last subplot if not needed
    if len(categories) < 8:
        axes[1, 3].remove()
    
    plt.tight_layout()
    plt.savefig('shared_data/runs_claude/20250727_212029/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_statistical_analysis_plots(df):
    """Create plots for statistical analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Statistical Analysis of Claude Results', fontsize=16, fontweight='bold')
    
    # 1. ANOVA Results Visualization
    ax1 = axes[0, 0]
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality']
    
    # Calculate F-statistics (simplified)
    f_stats = []
    for category in categories:
        groups = [df[df['type'] == valence][category] for valence in ['Neutral', 'Supportive', 'Threatening']]
        # Simplified F-statistic calculation
        overall_mean = df[category].mean()
        between_group_var = sum(len(g) * ((g.mean() - overall_mean) ** 2) for g in groups) / 2
        within_group_var = sum(sum((x - g.mean()) ** 2 for x in g) for g in groups) / (len(df) - 3)
        f_stat = between_group_var / within_group_var if within_group_var > 0 else 0
        f_stats.append(f_stat)
    
    bars = ax1.bar(range(len(categories)), f_stats, color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Evaluation Categories')
    ax1.set_ylabel('F-Statistic')
    ax1.set_title('F-Statistics by Category (ANOVA)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add significance line
    ax1.axhline(y=3.15, color='red', linestyle='--', alpha=0.7, label='F(2,57) = 3.15 (α=0.05)')
    ax1.legend()
    
    # 2. Correlation Matrix
    ax2 = axes[0, 1]
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality', 'total_score']
    
    corr_matrix = df[categories].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
    ax2.set_title('Correlation Matrix')
    
    # 3. Score Distribution by Valence
    ax3 = axes[1, 0]
    valence_order = ['Neutral', 'Supportive', 'Threatening']
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    for i, valence in enumerate(valence_order):
        data = df[df['type'] == valence]['total_score']
        ax3.hist(data, alpha=0.6, label=valence, color=colors[i], bins=15, density=True)
    
    ax3.set_xlabel('Total Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution by Valence (Normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Consistency
    ax4 = axes[1, 1]
    consistency_data = df.groupby('type')['total_score'].agg(['mean', 'std']).reset_index()
    consistency_data['cv'] = consistency_data['std'] / consistency_data['mean']  # Coefficient of variation
    
    bars = ax4.bar(consistency_data['type'], consistency_data['cv'], 
                   color=['#2E8B57', '#4682B4', '#CD5C5C'], alpha=0.8)
    ax4.set_xlabel('Valence Type')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Performance Consistency (Lower = More Consistent)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shared_data/runs_claude/20250727_212029/statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard."""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Claude Study Results - Comprehensive Dashboard', fontsize=20, fontweight='bold')
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Overall Performance Summary (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    valence_order = ['Neutral', 'Supportive', 'Threatening']
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    means = df.groupby('type')['total_score'].mean()
    stds = df.groupby('type')['total_score'].std()
    
    bars = ax1.bar(valence_order, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Total Score')
    ax1.set_title('Overall Performance by Valence')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Category Performance (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    categories = ['relevance_task', 'factual_accuracy', 'coherence_structure', 
                  'depth_insight', 'linguistic_quality', 'instruction_sensitivity', 'creativity_originality']
    
    category_means = df.groupby('type')[categories].mean()
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, valence in enumerate(valence_order):
        values = category_means.loc[valence].values
        ax2.bar(x + i*width, values, width, label=valence, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Evaluation Categories')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Category Performance by Valence')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top Topics Performance (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    top_topics = df.groupby('topic')['total_score'].mean().nlargest(8)
    
    bars = ax3.barh(range(len(top_topics)), top_topics.values, color='skyblue', alpha=0.8)
    ax3.set_yticks(range(len(top_topics)))
    ax3.set_yticklabels(top_topics.index, fontsize=9)
    ax3.set_xlabel('Average Total Score')
    ax3.set_title('Top 8 Topics by Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Score Distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    for i, valence in enumerate(valence_order):
        data = df[df['type'] == valence]['total_score']
        ax4.hist(data, alpha=0.6, label=valence, color=colors[i], bins=12, density=True)
    
    ax4.set_xlabel('Total Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Score Distribution by Valence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Heatmap (bottom left)
    ax5 = fig.add_subplot(gs[2:, :2])
    topic_valence_means = df.groupby(['topic', 'type'])['total_score'].mean().unstack()
    
    sns.heatmap(topic_valence_means, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5)
    ax5.set_title('Topic-Valence Performance Heatmap')
    ax5.set_xlabel('Valence Type')
    ax5.set_ylabel('Topic')
    
    # 6. Statistical Summary (bottom right)
    ax6 = fig.add_subplot(gs[2:, 2:])
    ax6.axis('off')
    
    # Create summary statistics table
    summary_stats = df.groupby('type')['total_score'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    
    table_data = []
    for valence in valence_order:
        row = [valence, f"{summary_stats.loc[valence, 'mean']:.2f}", 
               f"{summary_stats.loc[valence, 'std']:.2f}",
               f"{summary_stats.loc[valence, 'min']:.2f}",
               f"{summary_stats.loc[valence, 'max']:.2f}",
               int(summary_stats.loc[valence, 'count'])]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data, 
                     colLabels=['Valence', 'Mean', 'Std Dev', 'Min', 'Max', 'Count'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('shared_data/runs_claude/20250727_212029/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all graphs."""
    print("Loading Claude study data...")
    df = load_data()
    
    print("Creating valence comparison plots...")
    create_valence_comparison_plots(df)
    
    print("Creating topic analysis plots...")
    create_topic_analysis_plots(df)
    
    print("Creating detailed category plots...")
    create_detailed_category_plots(df)
    
    print("Creating statistical analysis plots...")
    create_statistical_analysis_plots(df)
    
    print("Creating comprehensive dashboard...")
    create_summary_dashboard(df)
    
    print("All graphs have been created and saved!")
    print("Files saved in: shared_data/runs_claude/20250727_212029/")

if __name__ == "__main__":
    main() 