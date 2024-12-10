import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors

lines_df = pd.read_csv('Components/NewLines.csv')
arcs_df = pd.read_csv('Components/NewArcs.csv')

def calculate_endpoint(row):
    if row['direction'] == '+X':
        return row['x'] - row['length'], row['y']
    elif row['direction'] == '-X':
        return row['x'] + row['length'], row['y']
    elif row['direction'] == '+Y':
        return row['x'], row['y'] - row['length']
    elif row['direction'] == '-Y':
        return row['x'], row['y'] + row['length']

def plot_line(row, highlight=False):
    x_start, y_start = row['x'], row['y']
    x_end, y_end = calculate_endpoint(row)

    color = 'green' if highlight else 'blue'
    linewidth = 3 if highlight else 1
    alpha = 0.7 if highlight else 0.4 
    line, = plt.plot([-x_start, -x_end], [-y_start, -y_end], marker='o', color=color, linewidth=linewidth, alpha=alpha)
    return line

def plot_arc(row, highlight=False):
    theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
    if row['Rotation'] == 'CW':
        theta = np.flip(theta)

    x_arc = row['x'] + row['radius'] * np.cos(theta)
    y_arc = row['y'] + row['radius'] * np.sin(theta)

    color = 'green' if highlight else 'red'
    linewidth = 3 if highlight else 1
    alpha = 0.7 if highlight else 0.4 
    arc, = plt.plot(-x_arc, -y_arc, color=color, linewidth=linewidth, alpha=alpha)
    return arc

def plot_with_highlights(highlight_lines=None, highlight_arcs=None):
    plt.figure(figsize=(10, 10))

    plotted_objects = []
    labels = []

    for _, row in lines_df.iterrows():
        highlight = row['index'] in highlight_lines if highlight_lines else False
        line = plot_line(row, highlight)
        plotted_objects.append(line)
        labels.append(f"Line {row['index']}")

    for _, row in arcs_df.iterrows():
        highlight = row['index'] in highlight_arcs if highlight_arcs else False
        arc = plot_arc(row, highlight)
        plotted_objects.append(arc)
        labels.append(f"Arc {row['index']}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lines and Arcs Plot from CSV Data')
    plt.grid(True)

    # Enable hover labels with mplcursors
    cursor = mplcursors.cursor(plotted_objects, hover=True)

    # Set the hover labels to show the respective label
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels[plotted_objects.index(sel.artist)])

    # Show the plot
    plt.show()

highlight_lines = []  # Replace with the actual line to highlight 
highlight_arcs = []  # Replace with the actual arc to highlight
plot_with_highlights(highlight_lines=highlight_lines, highlight_arcs=highlight_arcs)
