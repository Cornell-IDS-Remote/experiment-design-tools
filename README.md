# Map Processing and Visualization Tools

This repository contains tools and scripts for processing, analyzing, and visualizing map data consisting of lines and arcs. It is structured into three main directories:

1. **`lab_tools/`**: Tools used for interactive and automated operations in the lab.
2. **`helper_code/`**: Code for data generation and helper functions.
3. **`generating_map_testing/`**: Scripts for generating and testing map visualizations.

---

## Directory Structure

### `lab_tools/`
Contains tools for interactive visualization and map testing:
- **`hover_to_show_label_tool.py`**  
  Adds hover functionality to display labels for lines and arcs in the visualization.
- **`path_tool_labels.py`**  
  Interactive tool for selecting paths and toggling labels dynamically.
- **`path_tool_points.py`**  
  Enhances the interactivity by allowing users to mark and track specific points along the paths.

### `helper_code/`
Contains utilities for generating and processing data:
- **`gen_nodes_path.py`**  
  Processes node connections and paths from arcs and lines.
- **`gen_processed_data.py`**  
  Converts raw line and arc data into a processed format with calculated endpoints and tangents.

### `generating_map_testing/`
Contains scripts for visualizing and testing map generation:
- **`draw_arcs.py`**  
  Plots arcs based on input data from CSV files.
- **`draw_lines.py`**  
  Plots lines with direction and length information.
- **`draw_basic.py`**  
  Combines lines and arcs into a unified plot for visualization.

---

## Prerequisites

- **Python 3.7+**
- **Libraries**:
  - `matplotlib`
  - `pandas`
  - `numpy`
  - `mplcursors` (required for `hover_to_show_label_tool.py`)
