#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
import os
import matplotlib.pyplot as plt

def get_output_filename(base_name="plot_output.png"):
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    i = 1
    while os.path.exists(f"{name}_{i}{ext}"):
        i += 1
    return f"{name}_{i}{ext}"

def parse_log_file(filepath):
    # Regex to match lines like:
    # 200000 sfens, 7553 sfens/second, draw rate 54%, at Sat Feb 21 01:15:04 2026
    pattern = re.compile(r"(\d+) sfens, (\d+) sfens/second,.* at (.*)$")
    
    times = []
    total_fens = []
    fens_per_sec = []
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                fens = int(match.group(1))
                rate = int(match.group(2))
                time_str = match.group(3).strip()
                
                # Parse datetime
                # Format: Feb 21 01:15:04 2026
                dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
                
                total_fens.append(fens)
                fens_per_sec.append(rate)
                times.append(dt)
                
    return times, total_fens, fens_per_sec

def main():
    parser = argparse.ArgumentParser(description="Plot fens generation from log file")
    parser.add_argument("logfile", help="Path to the log file (e.g. gen_gardner_d4.log)")
    # add an optional overwrite flag
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()
    
    times, total_fens, fens_per_sec = parse_log_file(args.logfile)
    
    if not times:
        print("No data found in the log file.")
        return
        
    # Convert absolute times to relative minutes from the start
    start_time = times[0]
    relative_times = [(t - start_time).total_seconds() / 60.0 for t in times]
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Total Generated FENs', color=color1)
    ax1.plot(relative_times, total_fens, color=color1, marker='o', label='Total FENs')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)
    
    # Second y-axis for fens per second
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('FENs per second', color=color2)
    ax2.plot(relative_times, fens_per_sec, color=color2, marker='x', linestyle='--', label='FENs / sec')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    plt.title(f"FEN Generation Progress\n({args.logfile})")
    plt.tight_layout()
    
    # Save the plot
    output_png = get_output_filename("plot_output.png") if not args.overwrite else "plot_output.png"
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")
    
    # Optional: If you run this in an environment with a display, you can uncomment the next line
    # plt.show()

if __name__ == "__main__":
    main()
