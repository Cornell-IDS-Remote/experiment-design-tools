import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors
from collections import deque

# Load the processed CSV files for lines, arcs, and paths
lines_df = pd.read_csv('Processed_Lines.csv')
arcs_df = pd.read_csv('Processed_Arcs.csv')
nodes_paths_df = pd.read_csv('nodes_paths.csv')

# Dictionary to store the toggle state of each line and arc
toggle_state = {}
selected_segments = []
already_visited = set()  # Set to track already visited segments

# Function to calculate the endpoint of a line based on direction and length
def calculate_endpoint(row):
    if row['direction'] == '+X':
        return row['x'] - row['length'], row['y']
    elif row['direction'] == '-X':
        return row['x'] + row['length'], row['y']
    elif row['direction'] == '+Y':
        return row['x'], row['y'] - row['length']
    elif row['direction'] == '-Y':
        return row['x'], row['y'] + row['length']

# Function to calculate the length of a line
def calculate_line_length(row):
    x_start, y_start = row['x'], row['y']
    x_end, y_end = calculate_endpoint(row)
    return np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)

# Function to calculate the length of an arc
def calculate_arc_length(row):
    angle_diff = np.abs(row['angleEnd'] - row['angleStart'])
    return angle_diff * row['radius']

# Function to plot a line with transparency (alpha) and set initial thickness
def plot_line(row, color='blue', linewidth=1, alpha=0.5):
    x_start, y_start = row['x'], row['y']
    x_end, y_end = calculate_endpoint(row)

    # Plot the line and set the label
    line, = plt.plot([-x_start, -x_end], [-y_start, -y_end], color=color, linewidth=linewidth, alpha=alpha)
    line.set_label(f"Line {row['index']}")
    return line

# Function to plot an arc with transparency (alpha) and set initial thickness
def plot_arc(row, color='red', linewidth=1, alpha=0.5):
    theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
    if row['Rotation'] == 'CW':
        theta = np.flip(theta)

    x_arc = row['x'] + row['radius'] * np.cos(theta)
    y_arc = row['y'] + row['radius'] * np.sin(theta)

    # Plot the arc and set the label
    arc, = plt.plot(-x_arc, -y_arc, color=color, linewidth=linewidth, alpha=alpha)
    arc.set_label(f"Arc {row['index']}")
    return arc

# Plot everything (lines and arcs) with initial opacity and thin lines
def plot_all():
    plt.figure(figsize=(10, 10))

    # Plot lines
    for _, row in lines_df.iterrows():
        toggle_state[row['index']] = plot_line(row)
    
    # Plot arcs
    for _, row in arcs_df.iterrows():
        toggle_state[row['index']] = plot_arc(row)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interactive Plot of Lines and Arcs')
    plt.grid(True)

# Function to handle clicks and toggle color (with overlap check)
def on_click(event):
    overlapping_segments = []

    # Check for overlapping lines and arcs
    for idx, line in toggle_state.items():
        if line.contains(event)[0]:
            overlapping_segments.append(idx)

    # If there are multiple overlapping segments, select the shorter one
    if overlapping_segments:
        shortest_segment = find_shortest_segment(overlapping_segments)
        
        # If the segment has already been visited, ignore the click
        if shortest_segment in already_visited:
            print(f"Segment {shortest_segment} has already been visited, ignoring click.")
            return

        # If there are previous selected segments, try to extend the path
        if selected_segments:
            last_selected = selected_segments[-1]
            path = find_bfs_path(last_selected, shortest_segment)
            if path:
                print(path)
                # Color the found path and add segments to already visited set
                color_path_segments(path)
                already_visited.update(path)  # Mark path segments as visited
                selected_segments.append(shortest_segment)  # Add to selected segments
            else:
                print("No valid path found, ignoring click.")
                return
        else:
            # If no previous segments are selected, just add the segment
            selected_segments.append(shortest_segment)
            already_visited.add(shortest_segment)
            line = toggle_state[shortest_segment]
            line.set_color('green')
            line.set_linewidth(4)

        plt.draw()

# Function to find the shortest segment among overlapping segments
def find_shortest_segment(overlapping_segments):
    shortest_segment = None
    shortest_length = float('inf')

    for segment in overlapping_segments:
        if segment in lines_df['index'].values:
            row = lines_df[lines_df['index'] == segment].iloc[0]
            length = calculate_line_length(row)
        else:
            row = arcs_df[arcs_df['index'] == segment].iloc[0]
            length = calculate_arc_length(row)
        
        if length < shortest_length:
            shortest_length = length
            shortest_segment = segment

    return shortest_segment

# Function to reset the plot (deselect all) on key press
def on_key_press(event):
    if event.key == 'r':  # Reset when 'r' is pressed
        # Reset all lines and arcs to their original colors and thickness
        for idx, line in toggle_state.items():
            line.set_color('blue' if idx in lines_df['index'].values else 'red')
            line.set_linewidth(1)  # Reset thickness
        plt.draw()
        selected_segments.clear()
        already_visited.clear()
        print("Selection reset.")

# Function to find the path using BFS
def find_bfs_path(start, end):
    # Build the paths dictionary from the CSV
    paths_dict = {}
    for _, row in nodes_paths_df.iterrows():
        paths_dict[row['id']] = eval(row['paths'])  # Convert string to dictionary

    # BFS setup
    queue = deque([(start, None)])  # (current node, came from)
    visited = set(already_visited)  # Initialize visited with already visited nodes
    parent_map = {start: None}  # To reconstruct the path

    while queue:
        current, came_from = queue.popleft()
        visited.add(current)

        if current == end:
            return reconstruct_path(parent_map, start, end)

        # Get the paths for this node
        if current in paths_dict:
            if current == start:
                # For the starting node, we can leave from any path
                for exit_segment, next_segments in paths_dict[current].items():
                    for next_segment in next_segments:
                        if next_segment not in visited:
                            parent_map[next_segment] = current
                            queue.append((next_segment, current))
            else:
                # For other nodes, restrict movement based on the exit segment
                if came_from in paths_dict[current]:
                    for next_segment in paths_dict[current][came_from]:
                        if next_segment not in visited:
                            parent_map[next_segment] = current
                            queue.append((next_segment, current))

    return None  # No path found

def reconstruct_path(parent_map, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent_map[current]
    path.reverse()
    return path

# Function to color the segments that make up the found path, and make them thicker
def color_path_segments(path):
    for segment in path:
        if segment in toggle_state:
            toggle_state[segment].set_color('yellow')  # Change the path segments to yellow
            toggle_state[segment].set_linewidth(4)  # Make the path thicker for better visualization
    plt.draw()

plot_all()

fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()
