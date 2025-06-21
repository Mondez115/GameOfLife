import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cols = ["experiment", "accelerator", "size_x", "size_y", "double_dimension", "block_size", "cells_per_sec"]
results = pd.read_csv('results.csv', names=cols, header=None)

results['grid_size'] = results['size_x'] * results['size_y']

experiments = results["experiment"].unique()
accelerators = results["accelerator"].unique()

print(f"Experiments found: {experiments}")
print(f"Accelerators found: {accelerators}")

plt.figure(figsize=(10, 6))
exp1_results = results[results["experiment"] == "EXP1"]
for accelerator in accelerators:
    acc_results = exp1_results[exp1_results["accelerator"] == accelerator]
    if not acc_results.empty:
        acc_results = acc_results.sort_values('grid_size')
        plt.plot(acc_results["grid_size"], acc_results["cells_per_sec"], 
                marker='o', label=f"{accelerator}")

plt.xlabel("Grid Size (Total Cells) - Log Scale")
plt.ylabel("Cells Per Second")
plt.title("EXP1: Performance vs Grid Size (Log Scale)")
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.show()

plt.figure(figsize=(10, 6))
exp2_results = results[results["experiment"] == "EXP2"]
for accelerator in accelerators:
    acc_results = exp2_results[exp2_results["accelerator"] == accelerator]
    if not acc_results.empty:
        acc_results = acc_results.sort_values('grid_size')
        plt.plot(acc_results["grid_size"], acc_results["cells_per_sec"], 
                marker='o', label=f"{accelerator}")

plt.xlabel("Grid Size (Total Cells) - Log Scale")
plt.ylabel("Cells Per Second")
plt.title("EXP2: CUDA vs OpenCL Performance (Log Scale)")
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.show()

plt.figure(figsize=(10, 6))
exp3_results = results[results["experiment"] == "EXP3"]
for accelerator in accelerators:
    acc_results = exp3_results[exp3_results["accelerator"] == accelerator]
    if not acc_results.empty:
        acc_results = acc_results.sort_values('block_size')
        plt.plot(acc_results["block_size"], acc_results["cells_per_sec"], 
                marker='o', label=f"{accelerator}")

plt.xlabel("Block Size")
plt.ylabel("Cells Per Second")
plt.title("EXP3: Performance vs Block Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.show()

plt.figure(figsize=(10, 6))
exp4_results = results[results["experiment"] == "EXP4"]
for accelerator in accelerators:
    for double_dim in [0, 1]:
        acc_results = exp4_results[
            (exp4_results["accelerator"] == accelerator) & 
            (exp4_results["double_dimension"] == double_dim)
        ]
        if not acc_results.empty:
            acc_results = acc_results.sort_values('block_size')
            dim_label = "2D" if double_dim == 1 else "1D"
            plt.plot(acc_results["block_size"], acc_results["cells_per_sec"], 
                    marker='o', label=f"{accelerator} ({dim_label})")

plt.xlabel("Block Size - Log Scale")
plt.ylabel("Cells Per Second")
plt.title("EXP4: Single vs Double Dimension (Log Scale)")
plt.xscale('log')
plt.xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.show()
