
'''
After using the benchmark_dataset_size.py script, we have a list of logs for each dataset size.
This script parses the logs and plots the results in two graphs, allowing us to compare
the performance of the baseline model with different dataset sizes, this way we can observe
if the model gets better depending on the dataset size or the contents of the dataset (different depths)
'''


import os
import re
import matplotlib.pyplot as plt

def parse_logs(directory):
    results = []
    
    if not os.path.exists(directory):
        return results

    for filename in os.listdir(directory):
        if filename.startswith("logs_") and filename.endswith(".txt"):
            size_str = filename[len("logs_"):-len(".txt")]
            try:
                size = int(size_str)
            except ValueError:
                continue
                
            filepath = os.path.join(directory, filename)
            move_acc = 0.0
            res_acc = 0.0
            
            with open(filepath, 'r') as f:
                for line in f:
                    if "Best move accuracy:" in line:
                        m = re.search(r"([\d\.]+)%", line)
                        if m: move_acc = float(m.group(1))
                    elif "Best result accuracy:" in line:
                        m = re.search(r"([\d\.]+)%", line)
                        if m: res_acc = float(m.group(1))
                        
            if move_acc > 0 and res_acc > 0:
                results.append((size, move_acc, res_acc))
                
    # Sort by size
    results.sort(key=lambda x: x[0])
    return results

def main():
    base_dir = "src/benchmarks"
    
    runs = {
        "random_baseline": "Random Baseline",
        "d2": "Depth 2",
        "d2_bn_drop": "Depth 2 + Batchnorm + Dropout",
        "d3": "Depth 3",
        "d4": "Depth 4",
        "big1": "Merged (d4+d3+d2)"
    }
    
    # Colors and markers for clarity
    styles = {
        "random_baseline": {"color": "tab:gray", "marker": "x", "linestyle": "--"},
        "d2": {"color": "tab:blue", "marker": "o"},
        "d2_bn_drop": {"color": "tab:purple", "marker": "P"},
        "d3": {"color": "tab:orange", "marker": "s"},
        "d4": {"color": "tab:green", "marker": "^"},
        "big1": {"color": "tab:red", "marker": "D"},
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for run_dir, label in runs.items():
        full_dir = os.path.join(base_dir, run_dir)
        data = parse_logs(full_dir)
        
        if not data:
            print(f"Warning: No data found for {run_dir}")
            continue
            
        sizes = [d[0] for d in data]
        move_accs = [d[1] for d in data]
        res_accs = [d[2] for d in data]
        
        style = styles[run_dir]
        linestyle = style.get("linestyle", "-")
        
        # Plot Move Accuracy
        ax1.plot(sizes, move_accs, label=label, color=style["color"], marker=style["marker"], markersize=6, linewidth=2, linestyle=linestyle)
        
        # Plot Result Accuracy
        ax2.plot(sizes, res_accs, label=label, color=style["color"], marker=style["marker"], markersize=6, linewidth=2, linestyle=linestyle)

    # Styling ax1 (Move Accuracy)
    ax1.set_xscale('log')
    ax1.set_xlabel('Tamaño del Dataset (instancias)', fontsize=12)
    ax1.set_ylabel('Move Accuracy (%)', fontsize=12)
    ax1.set_title('Move Accuracy vs Datos de Entrenamiento', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Styling ax2 (Result Accuracy)
    ax2.set_xscale('log')
    ax2.set_xlabel('Tamaño del Dataset (instancias)', fontsize=12)
    ax2.set_ylabel('Result Accuracy (%)', fontsize=12)
    ax2.set_title('Result Accuracy vs Datos de Entrenamiento', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(base_dir, "overlapped_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n[*] Gráfica overlapped guardada con éxito en {output_path}")

if __name__ == "__main__":
    main()
