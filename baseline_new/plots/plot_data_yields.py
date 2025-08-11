
#!/usr/bin/env python3
"""
Plot data yields for generalization mappings across model sizes.

This script creates a grid plot showing the percentage of samples where models lied
(did_lie=true) for each generalization category across different model sizes.

It reads lie_taxonomy.csv for category mappings.

Creates a grid:
- Each column in lie_taxonomy.csv (Map 1, Map 2, etc.) is a "row" in the grid plot.
- X-axis: Model sizes (4b, 12b, 27b)
- Y-axis: Percentage of samples where did_lie=true
"""

import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# Paths
TAXONOMY_FILE = "/Users/dipikakhullar/Desktop/lie-detector/baseline/lie_taxonomy_clean3.csv"
BUCKET_NAME = 'dipika-lie-detection-data'
PREFIX = 'processed-data-gemma/'

# Model size mapping
MODEL_SIZE_MAPPING = {
    'gemma-3-4b-it': '4b',
    'gemma-3-12b-it': '12b',
    'gemma-3-27b-it': '27b',
    'google_gemma-3-4b-it': '4b',
    'google_gemma-3-12b-it': '12b',
    'google_gemma-3-27b-it': '27b',
    'openrouter/google/gemma-3-4b-it': '4b',
    'openrouter/google/gemma-3-12b-it': '12b',
    'openrouter/google/gemma-3-27b-it': '27b'
}

# Configure S3 client
s3_client = boto3.client('s3')

def build_generalization_maps(df: pd.DataFrame, map_columns):
    """Build generalization mappings from CSV for each specified column."""
    mappings = {}
    for map_col in map_columns:
        mapping = defaultdict(list)
        for _, row in df.iterrows():
            task = str(row['Task']).strip().lower()
            category = str(row[map_col]).strip().lower()
            if task and category and category != 'nan':
                # Handle "Falsification / Exaggeration" by adding to both categories
                if '/' in category:
                    categories = [cat.strip() for cat in category.split('/')]
                    for cat in categories:
                        mapping[cat].append(task.replace('_', '-'))
                else:
                    mapping[category].append(task.replace('_', '-'))
        mappings[map_col] = dict(mapping)
    return mappings


def get_task_files_from_s3():
    """Get all task files from S3 bucket."""
    task_files = defaultdict(list)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.jsonl') and 'deduplicated' in key:
                    parts = key.split('/')
                    if len(parts) >= 3:
                        task_name = parts[1]
                        task_files[task_name].append(key)
    return task_files

def load_task_data(task_files):
    """Load data for each task and model."""
    task_data = defaultdict(lambda: defaultdict(list))
    for task_name, files in task_files.items():
        for file_key in files:
            filename = file_key.split('/')[-1]
            model_part = filename.replace('deduplicated_', '').replace('.jsonl', '')
            model = model_part.split('_')[-1]
            model_size = MODEL_SIZE_MAPPING.get(model)
            if not model_size:
                continue
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            samples = []
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            task_data[task_name][model_size].extend(samples)
    return task_data

def calculate_lie_percentage(samples):
    """Calculate percentage of samples where did_lie=true."""
    if not samples:
        return 0.0
    lie_count = sum(1 for sample in samples if sample.get('did_lie', False))
    return (lie_count / len(samples)) * 100

def aggregate_by_category(task_data, generalization_map):
    """Aggregate data by generalization category and model size."""
    category_data = defaultdict(lambda: defaultdict(list))
    for category, task_names in generalization_map.items():
        for task_name in task_names:
            if task_name in task_data:
                for model_size, samples in task_data[task_name].items():
                    lie_percentage = calculate_lie_percentage(samples)
                    category_data[category][model_size].append(lie_percentage)
    # Average percentages
    result = {}
    for category, model_data in category_data.items():
        result[category] = {}
        for model_size, percentages in model_data.items():
            result[category][model_size] = sum(percentages) / len(percentages)
    return result

def create_grid_plot(map_data):
    """Create grid plots for each Map column."""
    rows = len(map_data)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 5 * rows))
    fig.suptitle('Lie Percentages by Category and Model Size', fontsize=16, fontweight='bold')
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    if rows == 1:
        axes = [axes]
    for ax, (map_name, category_data) in zip(axes, map_data.items()):
        for i, (category, data) in enumerate(category_data.items()):
            percentages = [data.get(size, 0) for size in model_sizes]
            ax.plot(model_sizes, percentages, label=category.title(), marker='o', color=colors[i % len(colors)])
        ax.set_title(map_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Lie Percentage (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True)
    plt.xlabel('Model Size')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('data_yields', exist_ok=True)
    output_path = 'data_yields/grid_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Grid plot saved to: {output_path}")





def create_grid_bar_plots(all_maps, task_data):
    """Create grid bar plots for all columns/categories."""
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    for map_name, generalization_map in all_maps.items():
        num_categories = len(generalization_map)
        fig, axes = plt.subplots(1, num_categories, figsize=(5 * num_categories, 6))
        fig.suptitle(f'{map_name}: Percentage of Samples Where Model Lied (did_lie=true)',
                     fontsize=16, fontweight='bold')

        if num_categories == 1:
            axes = [axes]

        for i, (category, tasks) in enumerate(generalization_map.items()):
            ax = axes[i]
            percentages = []
            for size in model_sizes:
                all_samples = []
                for task in tasks:
                    if task in task_data and size in task_data[task]:
                        all_samples.extend(task_data[task][size])
                percentage = calculate_lie_percentage(all_samples)
                percentages.append(percentage)

            bars = ax.bar(model_sizes, percentages, color=colors[i % len(colors)], alpha=0.8)
            ax.set_title(category.title(), fontweight='bold')
            ax.set_ylabel('Lie Percentage (%)')
            ax.set_ylim(0, 100)
            for bar, percentage in zip(bars, percentages):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs('data_yields', exist_ok=True)
        output_path = f"data_yields/grid_bar_plot_{map_name.replace(' ', '_').lower()}.png"
        plt.savefig(output_path, dpi=300)
        print(f"Grid bar plot saved to: {output_path}")



# def create_combined_grid_bar_plot(all_maps, task_data):
#     """
#     Create one big grid bar plot for all columns/categories.
#     Rows = columns (Map 1, Map 2, etc.)
#     Columns = number of categories in each column.
#     """
#     model_sizes = ['4b', '12b', '27b']
#     colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

#     # Determine grid size
#     num_rows = len(all_maps)
#     max_cols = max(len(categories) for categories in all_maps.values())

#     # Create grid
#     fig, axes = plt.subplots(num_rows, max_cols,
#                              figsize=(5 * max_cols, 5 * num_rows),
#                              squeeze=False)
#     fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
#                  fontsize=18, fontweight='bold')

#     for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
#         categories = list(generalization_map.items())
#         for col_idx in range(max_cols):
#             ax = axes[row_idx][col_idx]

#             if col_idx < len(categories):
#                 category, tasks = categories[col_idx]
#                 percentages = []
#                 for size in model_sizes:
#                     all_samples = []
#                     for task in tasks:
#                         if task in task_data and size in task_data[task]:
#                             all_samples.extend(task_data[task][size])
#                     percentage = calculate_lie_percentage(all_samples)
#                     percentages.append(percentage)

#                 bars = ax.bar(model_sizes, percentages,
#                               color=colors[col_idx % len(colors)], alpha=0.8)
#                 ax.set_title(f'{map_name} - {category.title()}',
#                              fontsize=10, fontweight='bold')
#                 ax.set_ylim(0, 100)
#                 ax.set_ylabel('Lie %')
#                 ax.set_xlabel('Model Size')

#                 # Add value labels
#                 for bar, percentage in zip(bars, percentages):
#                     ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
#                             f'{percentage:.1f}%', ha='center', va='bottom',
#                             fontsize=8, fontweight='bold')
#             else:
#                 # Hide empty subplots
#                 ax.axis('off')

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     os.makedirs('data_yields', exist_ok=True)
#     output_path = "data_yields/combined_grid_bar_plot.png"
#     plt.savefig(output_path, dpi=300)
#     print(f"Combined grid bar plot saved to: {output_path}")



def create_combined_grid_bar_plot(all_maps, task_data):
    """
    Create one big grid bar plot for all columns/categories.
    Rows = columns (Map 1, Map 2, etc.)
    Columns = number of categories in each column.
    """
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    # Determine grid size
    num_rows = len(all_maps)
    max_cols = max(len(categories) for categories in all_maps.values())

    # Create grid
    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(5 * max_cols, 4 * num_rows),
                             squeeze=False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85, bottom=0.1)
    fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                 fontsize=18, fontweight='bold')

    for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
        categories = list(generalization_map.items())

        # Add row title
        fig.text(0.5, 1 - (row_idx / num_rows) - 0.02, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')

        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]

            if col_idx < len(categories):
                category, tasks = categories[col_idx]
                percentages = []
                for size in model_sizes:
                    all_samples = []
                    for task in tasks:
                        if task in task_data and size in task_data[task]:
                            all_samples.extend(task_data[task][size])
                    percentage = calculate_lie_percentage(all_samples)
                    percentages.append(percentage)

                bars = ax.bar(model_sizes, percentages,
                              color=colors[col_idx % len(colors)], alpha=0.8)

                # Set subplot title (just category)
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')

                ax.set_ylim(0, 100)
                ax.set_ylabel('Lie %')
                ax.set_xlabel('Model Size')

                # Add value labels
                for bar, percentage in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{percentage:.1f}%', ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
            else:
                # Hide empty subplots
                ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs('data_yields', exist_ok=True)
    output_path = "data_yields/combined_grid_bar_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Combined grid bar plot saved to: {output_path}")




def create_combined_grid_bar_plot(all_maps, task_data):
    """
    Create one big grid bar plot for all columns/categories.
    Rows = columns (Map 1, Map 2, etc.)
    Columns = number of categories in each column.
    """
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    # Determine grid size
    num_rows = len(all_maps)
    max_cols = max(len(categories) for categories in all_maps.values())

    # Create grid
    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(5 * max_cols, 4 * num_rows),
                             squeeze=False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, bottom=0.1)
    fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                 fontsize=18, fontweight='bold')

    for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
        categories = list(generalization_map.items())

        # Calculate proper vertical position for each row title
        # Position title above each row of subplots by using subplot positions
        # Get the position of the first subplot in this row to align the title
        subplot_pos = axes[row_idx][0].get_position()
        title_y = subplot_pos.y1 + 0.02  # Position slightly above the top of the subplot
        
        fig.text(0.5, title_y, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')

        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]

            if col_idx < len(categories):
                category, tasks = categories[col_idx]
                percentages = []
                for size in model_sizes:
                    all_samples = []
                    for task in tasks:
                        if task in task_data and size in task_data[task]:
                            all_samples.extend(task_data[task][size])
                    percentage = calculate_lie_percentage(all_samples)
                    percentages.append(percentage)

                bars = ax.bar(model_sizes, percentages,
                              color=colors[col_idx % len(colors)], alpha=0.8)

                # Set subplot title (just category)
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')

                ax.set_ylim(0, 100)
                ax.set_ylabel('Lie %')
                ax.set_xlabel('Model Size')

                # Add value labels
                for bar, percentage in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{percentage:.1f}%', ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
            else:
                # Hide empty subplots
                ax.axis('off')

    # Don't use tight_layout since we're manually positioning titles
    os.makedirs('data_yields', exist_ok=True)
    output_path = "data_yields/combined_grid_bar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined grid bar plot saved to: {output_path}")


def create_combined_grid_bar_plot(all_maps, task_data):
    """
    Create one big grid bar plot for all columns/categories.
    Rows = columns (Map 1, Map 2, etc.)
    Columns = number of categories in each column.
    """
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    # Determine grid size
    num_rows = len(all_maps)
    max_cols = max(len(categories) for categories in all_maps.values())

    # Create grid
    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(5 * max_cols, 4 * num_rows),
                             squeeze=False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, bottom=0.1)
    fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                 fontsize=18, fontweight='bold')

    for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
        categories = list(generalization_map.items())

        # Calculate proper vertical position for each row title
        # Position title above each row of subplots by using subplot positions
        # Get the position of the first subplot in this row to align the title
        subplot_pos = axes[row_idx][0].get_position()
        title_y = subplot_pos.y1 + 0.04  # Increased distance above the subplot
        
        fig.text(0.5, title_y, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')

        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]

            if col_idx < len(categories):
                category, tasks = categories[col_idx]
                percentages = []
                for size in model_sizes:
                    all_samples = []
                    for task in tasks:
                        if task in task_data and size in task_data[task]:
                            all_samples.extend(task_data[task][size])
                    percentage = calculate_lie_percentage(all_samples)
                    percentages.append(percentage)

                bars = ax.bar(model_sizes, percentages,
                              color=colors[col_idx % len(colors)], alpha=0.8)

                # Set subplot title (just category)
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')

                ax.set_ylim(0, 100)
                ax.set_ylabel('Lie %')
                ax.set_xlabel('Model Size')

                # Add value labels
                for bar, percentage in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{percentage:.1f}%', ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
            else:
                # Hide empty subplots
                ax.axis('off')

    # Don't use tight_layout since we're manually positioning titles
    os.makedirs('data_yields', exist_ok=True)
    output_path = "data_yields/combined_grid_bar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined grid bar plot saved to: {output_path}")
    # plt.show()



def create_combined_grid_bar_plot(all_maps, task_data):
    """
    Create one big grid bar plot for all columns/categories.
    Rows = columns (Map 1, Map 2, etc.)
    Columns = number of categories in each column.
    """
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    # Determine grid size
    num_rows = len(all_maps)
    max_cols = max(len(categories) for categories in all_maps.values())

    # Create grid
    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(5 * max_cols, 4 * num_rows),
                             squeeze=False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, bottom=0.1)
    fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                 fontsize=18, fontweight='bold')

    for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
        categories = list(generalization_map.items())

        # Calculate proper vertical position for each row title
        # Position title in the gap between rows, closer to the current row
        subplot_pos = axes[row_idx][0].get_position()
        
        if row_idx == 0:
            # For the first row, position above the subplots
            title_y = subplot_pos.y1 + 0.02
        else:
            # For subsequent rows, position in the gap but closer to current row
            # Get the bottom position of the previous row
            prev_subplot_pos = axes[row_idx-1][0].get_position()
            gap_size = subplot_pos.y1 - prev_subplot_pos.y0
            # Position title at 70% down from previous row (30% up from current row)
            title_y = prev_subplot_pos.y0 - (gap_size * 0.3)
        
        fig.text(0.5, title_y, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')

        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]

            if col_idx < len(categories):
                category, tasks = categories[col_idx]
                percentages = []
                for size in model_sizes:
                    all_samples = []
                    for task in tasks:
                        if task in task_data and size in task_data[task]:
                            all_samples.extend(task_data[task][size])
                    percentage = calculate_lie_percentage(all_samples)
                    percentages.append(percentage)

                bars = ax.bar(model_sizes, percentages,
                              color=colors[col_idx % len(colors)], alpha=0.8)

                # Set subplot title (just category)
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')

                ax.set_ylim(0, 100)
                ax.set_ylabel('Lie %')
                ax.set_xlabel('Model Size')

                # Add value labels
                for bar, percentage in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{percentage:.1f}%', ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
            else:
                # Hide empty subplots
                ax.axis('off')

    # Don't use tight_layout since we're manually positioning titles
    os.makedirs('data_yields', exist_ok=True)
    output_path = "data_yields/combined_grid_bar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined grid bar plot saved to: {output_path}")
    # plt.show()




    def create_combined_grid_bar_plot(all_maps, task_data):
        """
        Create one big grid bar plot for all columns/categories.
        Rows = columns (Map 1, Map 2, etc.)
        Columns = number of categories in each column.
        """
        model_sizes = ['4b', '12b', '27b']
        colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

        # Determine grid size
        num_rows = len(all_maps)
        max_cols = max(len(categories) for categories in all_maps.values())

        # Create grid
        fig, axes = plt.subplots(num_rows, max_cols,
                                figsize=(5 * max_cols, 4 * num_rows),
                                squeeze=False)

        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, bottom=0.1)
        fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                    fontsize=18, fontweight='bold')

        for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
            categories = list(generalization_map.items())

            # Calculate proper vertical position for each row title
            # Position title directly above each row, accounting for the subplot title space
            subplot_pos = axes[row_idx][0].get_position()
            # Position title above the subplot titles (which are above the actual plots)
            title_y = subplot_pos.y1 + 0.06  # More space to clear the subplot titles
            
            fig.text(0.5, title_y, map_name,
                    ha='center', va='center', fontsize=14, fontweight='bold')

            for col_idx in range(max_cols):
                ax = axes[row_idx][col_idx]

                if col_idx < len(categories):
                    category, tasks = categories[col_idx]
                    percentages = []
                    for size in model_sizes:
                        all_samples = []
                        for task in tasks:
                            if task in task_data and size in task_data[task]:
                                all_samples.extend(task_data[task][size])
                        percentage = calculate_lie_percentage(all_samples)
                        percentages.append(percentage)

                    bars = ax.bar(model_sizes, percentages,
                                color=colors[col_idx % len(colors)], alpha=0.8)

                    # Set subplot title (just category)
                    ax.set_title(category.replace('_', ' ').title(),
                                fontsize=10, fontweight='bold')

                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Lie %')
                    ax.set_xlabel('Model Size')

                    # Add value labels
                    for bar, percentage in zip(bars, percentages):
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                                f'{percentage:.1f}%', ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                else:
                    # Hide empty subplots
                    ax.axis('off')

        # Don't use tight_layout since we're manually positioning titles
        os.makedirs('data_yields', exist_ok=True)
        output_path = "data_yields/combined_grid_bar_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined grid bar plot saved to: {output_path}")


def create_combined_grid_bar_plot(all_maps, task_data):
    """
    Create one big grid bar plot for all columns/categories.
    Rows = columns (Map 1, Map 2, etc.)
    Columns = number of categories in each column.
    """
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    # Determine grid size
    num_rows = len(all_maps)
    max_cols = max(len(categories) for categories in all_maps.values())

    # Create grid
    fig, axes = plt.subplots(num_rows, max_cols,
                             figsize=(5 * max_cols, 4 * num_rows),
                             squeeze=False)

    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.90, bottom=0.1)
    fig.suptitle('Percentage of Samples Where Model Lied (did_lie=true)',
                 fontsize=18, fontweight='bold')

    for row_idx, (map_name, generalization_map) in enumerate(all_maps.items()):
        categories = list(generalization_map.items())

        # Calculate proper vertical position for each row title
        # Position title in the gap between the subplot title and the actual plot area
        subplot_pos = axes[row_idx][0].get_position()
        # Position title just above the subplot, but below any subplot titles
        title_y = subplot_pos.y1 + 0.02  # Very close to the top of the subplot
        
        fig.text(0.5, title_y, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')

        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]

            if col_idx < len(categories):
                category, tasks = categories[col_idx]
                percentages = []
                for size in model_sizes:
                    all_samples = []
                    for task in tasks:
                        if task in task_data and size in task_data[task]:
                            all_samples.extend(task_data[task][size])
                    percentage = calculate_lie_percentage(all_samples)
                    percentages.append(percentage)

                bars = ax.bar(model_sizes, percentages,
                              color=colors[col_idx % len(colors)], alpha=0.8)

                # Set subplot title (just category)
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')

                ax.set_ylim(0, 100)
                ax.set_ylabel('Lie %')
                ax.set_xlabel('Model Size')

                # Add value labels
                for bar, percentage in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{percentage:.1f}%', ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
            else:
                # Hide empty subplots
                ax.axis('off')

    # Don't use tight_layout since we're manually positioning titles
    os.makedirs('data_yields', exist_ok=True)
    output_path = "data_yields/combined_grid_bar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined grid bar plot saved to: {output_path}")
    # plt.show()


def main():
    print("Loading taxonomy...")
    taxonomy_df = pd.read_csv(TAXONOMY_FILE)
    # map_columns = ['Map 1', 'Map 2']  # Adjust if there are more
    map_columns = [col for col in taxonomy_df.columns if col.strip().lower() != 'task']
    mappings = build_generalization_maps(taxonomy_df, map_columns)
    print("Fetching task files from S3...")
    task_files = get_task_files_from_s3()
    print("Loading task data...")
    task_data = load_task_data(task_files)
    print("Aggregating and plotting data...")
    aggregated_data = {map_name: aggregate_by_category(task_data, generalization_map)
                       for map_name, generalization_map in mappings.items()}
    
    print("Creating line plots...")
    create_grid_plot(aggregated_data)
    print("Creating bar charts...")
    print("Creating grid bar plots...")
    create_grid_bar_plots(mappings, task_data)

    print("Creating combined grid bar plot...")
    create_combined_grid_bar_plot(mappings, task_data)



if __name__ == "__main__":
    main()
