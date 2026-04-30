import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_f1_scores(models, classes, output_dir, save_path):
    f1_scores = {model: [] for model in models}
    
    for model in models:
        json_path = os.path.join(output_dir, model, "classification_report.json")
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
    # Adjust width based on number of models so bars fit around the center
    width = 0.8 / len(models) 
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
    
    # Grayscale colors and patterns for better distinguishability in black and white
    colors = ['#404040', '#808080', '#C0C0C0', '#202020', '#A0A0A0', '#606060']
    hatches = ['', '//', '\\\\', 'xx', '..', 'oo']
    
    # Center the bars around the tick marks
    start_offset = - (len(models) - 1) * width / 2
    
    for i, (model, scores) in enumerate(f1_scores.items()):
        offset = start_offset + i * width
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        rects = ax.bar(x + offset, scores, width, label=model, color=color, edgecolor='black', hatch=hatch)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)
        
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison by Word Class and Model')
    ax.set_xticks(x, classes)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1) # Add some space above for labels
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize F1-Scores across different models.")
    parser.add_argument("--models", nargs='+', default=["M1", "M4", "M6"], help="List of models to compare")
    parser.add_argument("--classes", nargs='+', default=["N-ABS", "N-KON", "VB-T", "INTJ", "UNID"], help="List of word classes to compare")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Base directory where model outputs are stored")
    parser.add_argument("--save_path", type=str, default="f1_comparison.png", help="Path to save the generated graph")
    
    args = parser.parse_args()
    
    plot_f1_scores(args.models, args.classes, args.outputs_dir, args.save_path)
