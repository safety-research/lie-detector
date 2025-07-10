import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# Create the dataset
data = {
    'prompt_type': ['safety_concern', 'training_corruption', 'capability_hiding', 'safety_concern', 'training_corruption', 'capability_hiding'],
    'sandbagging': [0.39, 0.65, 0.27, 0.24, 0.37, 0.31],
    'baseline': [0.67, 0.81, 0.78, 0.39, 0.34, 0.34],
    'model': ['openrouter/openai/gpt-4o', 'openrouter/openai/gpt-4o',
              'openrouter/openai/gpt-4o', 'openrouter/meta-llama/llama-3.1-8b-instruct',
              'openrouter/meta-llama/llama-3.1-8b-instruct', 'openrouter/meta-llama/llama-3.1-8b-instruct']
}

df = pd.DataFrame(data)

# Clean up model names for better display
df['model_clean'] = df['model'].str.replace('openrouter/', '').str.replace('openai/', '').str.replace('meta-llama/', '')

# Calculate sandbagging effect (baseline - sandbagging)
df['sandbagging_effect'] = df['baseline'] - df['sandbagging']

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a single figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('Chess Sandbagging Detection Results', fontsize=16, fontweight='bold')

# Define conventional colors for models
model_colors = {
    'gpt-4o': '#1f77b4',  # Blue (conventional for OpenAI)
    'llama-3.1-8b-instruct': '#ff7f0e'  # Orange (conventional for Meta/Llama)
}

# Group data by prompt type for better organization
prompt_types = df['prompt_type'].unique()
models = df['model_clean'].unique()

# Calculate positions for candlesticks
n_prompts = len(prompt_types)
n_models = len(models)
width = 0.35
x_base = np.arange(n_prompts)

# Create candlestick chart
for i, prompt in enumerate(prompt_types):
    for j, model in enumerate(models):
        # Get data for this prompt type and model
        data = df[(df['prompt_type'] == prompt) & (df['model_clean'] == model)]
        
        if len(data) > 0:
            row = data.iloc[0]  # Should be only one row per combination
            baseline = row['baseline']
            sandbagging = row['sandbagging']
            
            # Position for this candlestick
            x_pos = i + (j - (n_models-1)/2) * width
            
            # Determine high and low values
            high = max(baseline, sandbagging)
            low = min(baseline, sandbagging)
            
            # Use model color
            color = model_colors.get(model, 'gray')
            
            # Draw the rectangle (candlestick body)
            rect = Rectangle((x_pos - width/2, low), width, high - low, 
                            facecolor=color, alpha=0.6, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Draw center line at baseline (thicker line)
            ax.hlines(baseline, x_pos - width/2, x_pos + width/2, 
                     colors='black', linewidth=3)
            
            # Add value labels
            ax.text(x_pos, high + 0.01, f'{high:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(x_pos, low - 0.01, f'{low:.2f}', ha='center', va='top', fontsize=8)

# Set up the plot
ax.set_xlabel('Prompt Type', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Candlestick: Baseline vs Sandbagging Performance by Model', fontsize=14)
ax.set_xticks(x_base)
ax.set_xticklabels(prompt_types, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Create legend for models
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=model_colors.get(model, 'gray'), alpha=0.6, label=model) 
    for model in models
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("\nOverall Results:")
print(f"Average Baseline Accuracy: {df['baseline'].mean():.3f}")
print(f"Average Sandbagging Accuracy: {df['sandbagging'].mean():.3f}")
print(f"Average Sandbagging Effect: {df['sandbagging_effect'].mean():.3f}")

print("\nBy Model:")
for model in models:
    model_data = df[df['model_clean'] == model]
    print(f"\n{model}:")
    print(f"  Baseline: {model_data['baseline'].mean():.3f}")
    print(f"  Sandbagging: {model_data['sandbagging'].mean():.3f}")
    print(f"  Effect: {model_data['sandbagging_effect'].mean():.3f}")

print("\nBy Prompt Type:")
for prompt in df['prompt_type'].unique():
    prompt_data = df[df['prompt_type'] == prompt]
    print(f"\n{prompt}:")
    print(f"  Baseline: {prompt_data['baseline'].mean():.3f}")
    print(f"  Sandbagging: {prompt_data['sandbagging'].mean():.3f}")
    print(f"  Effect: {prompt_data['sandbagging_effect'].mean():.3f}")