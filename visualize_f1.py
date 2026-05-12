import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

# Research-ready Premium Grayscale style
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#000000",
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "grid.linestyle":    "--",
    "legend.fontsize":   10,
    "legend.frameon":    True,
    "legend.edgecolor":  "#333333",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans", "Helvetica"],
    "savefig.dpi":       300, 
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def plot_f1_scores(models, classes, output_dir, save_path):
    f1_scores = {model: [] for model in models}
    
    for model in models:
        # Search for report in drive subfolders first
        drive_pattern = os.path.join(output_dir, f"drive*-{model.lower()}", f"classification_report_{model.lower()}.json")
        matches = glob.glob(drive_pattern)
        
        if matches:
            json_path = matches[0]
        else:
            # Fallback to direct path in output_dir
            json_path = os.path.join(output_dir, f"classification_report_{model.lower()}.json")
            
        if not os.path.exists(json_path):
            print(f"Warning: Data for model {model} not found at {json_path}. Filling with 0.")
            f1_scores[model] = [0.0] * len(classes)
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        for cls in classes:
            if cls in report:
                f1_scores[model].append(report[cls]['f1-score'])
            else:
                print(f"Warning: Class {cls} not found in model {model}'s report. Filling with 0.")
                f1_scores[model].append(0.0)
                
    # Plotting
    x = np.arange(len(classes))
    width = 0.8 / len(models) 
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Premium Grayscale colors and patterns
    colors = ['#333333', '#666666', '#999999', '#4d4d4d', '#b3b3b3', '#1a1a1a']
    hatches = ['', '///', '\\\\', 'xx', '..', '++']
    
    start_offset = - (len(models) - 1) * width / 2
    
    for i, (model, scores) in enumerate(f1_scores.items()):
        offset = start_offset + i * width
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        rects = ax.bar(x + offset, scores, width, label=model, 
                       color=color, edgecolor='black', hatch=hatch, linewidth=0.8)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9, weight='bold')
        
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('Word Class')
    ax.set_title('F1-Score Comparison by Word Class and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.1))
    ax.set_ylim(0, 1.1)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize F1-Scores across different models.")
    parser.add_argument("--models", nargs='+', default=["M1", "M4", "M6"], help="List of models to compare")
    parser.add_argument("--classes", nargs='+', default=["N-ABS", "N-KON", "VB-T", "INTJ", "UNID"], help="List of word classes to compare")
    parser.add_argument("--outputs_dir", type=str, default="training_result", help="Base directory where model outputs are stored")
    parser.add_argument("--save_path", type=str, default="./analyst/f1_comparison.png", help="Path to save the generated graph")
    
    args = parser.parse_args()
    
    plot_f1_scores(args.models, args.classes, args.outputs_dir, args.save_path)
