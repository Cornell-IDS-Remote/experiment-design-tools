import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque
from scipy.spatial import distance

lines_df = pd.read_csv('Processed_Lines.csv')
arcs_df = pd.read_csv('Processed_Arcs.csv')
nodes_paths_df = pd.read_csv('nodes_paths.csv')

toggle_state = {}
selected_segments = []
already_visited = set()  # Set to track already visited segments
closest_points = {}  # Dictionary to store closest points for clicked segments
closest_points1 = {}  # Dictionary to store closest points for clicked segments
plotted_points = []  # List to store plotted points for removal

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
    arc, = plt.plot(-x_arc, -y_arc, color=color, linewidth=linewidth, alpha=alpha)
    arc.set_label(f"Arc {row['index']}")
    return arc

# Modify plot_all to remove plt.figure()
def plot_all():
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

# Function to reconstruct the path from the BFS result
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


# Function to find the closest point on a line or arc to a given point
def find_closest_point(x_click, y_click, row):
    if row['index'] in lines_df['index'].values:

        x_click = -x_click
        y_click = -y_click

        # Get the start and end points of the line segment
        x_start, y_start = row['x'], row['y']
        x_end, y_end = calculate_endpoint(row)

        # Line segment vector
        line_dx = x_end - x_start
        line_dy = y_end - y_start

        # Vector from start to the click point
        click_dx = x_click - x_start
        click_dy = y_click - y_start

        # Project the click point onto the line segment using the dot product
        line_length_squared = line_dx ** 2 + line_dy ** 2
        t = (click_dx * line_dx + click_dy * line_dy) / line_length_squared

        # Clamp t to the range [0, 1] to stay within the segment bounds
        t = max(0, min(1, t))

        # Calculate the closest point on the line segment
        closest_x = x_start + t * line_dx
        closest_y = y_start + t * line_dy

        return -closest_x, -closest_y

    else:  # For arcs, keep existing behavior
        theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
        x_arc = row['x'] + row['radius'] * np.cos(theta)
        y_arc = row['y'] + row['radius'] * np.sin(theta)
        line_points = np.column_stack((x_arc, y_arc))
        
        # Calculate distances to find the closest point on the arc
        distances = distance.cdist([(x_click, y_click)], line_points)
        min_index = distances.argmin()
        return line_points[min_index]

# Function to reset the plot (deselect all) on key press
def on_key_press(event):
    if event.key == 'r':  # Reset when 'r' is pressed
        for idx, line in toggle_state.items():
            line.set_color('blue' if idx in lines_df['index'].values else 'red')
            line.set_linewidth(1)  # Reset thickness
        
        for point in plotted_points:
            point.remove()
        plotted_points.clear() 
        
        plt.draw()
        selected_segments.clear()
        already_visited.clear()
        closest_points.clear() 
        print("Selection reset.")
        
def on_click(event):

   
    
    overlapping_segments = []
    for idx, line in toggle_state.items():
        if line.contains(event)[0]:
            overlapping_segments.append(idx)
    if overlapping_segments:
        shortest_segment = find_shortest_segment(overlapping_segments)
        if shortest_segment in already_visited:
            print(f"Segment {shortest_segment} has already been visited, ignoring click.")
            return
        if selected_segments:
            last_selected = selected_segments[-1]
            path = find_bfs_path(last_selected, shortest_segment)
            if path:
                print(path)
                color_path_segments(path)
                already_visited.update(path)
                selected_segments.append(shortest_segment)
                save_closest_points(path, event.xdata, event.ydata)
                closest_points1[shortest_segment] = find_closest_point(event.xdata, event.ydata, lines_df.loc[lines_df['index'] == shortest_segment].iloc[0])

            else:
                print("No valid path found, ignoring click.")
                return
        else:
            selected_segments.append(shortest_segment)
            already_visited.add(shortest_segment)
            line = toggle_state[shortest_segment]
            line.set_color('green')
            line.set_linewidth(4)
            
            closest_points[shortest_segment] = find_closest_point(event.xdata, event.ydata, lines_df.loc[lines_df['index'] == shortest_segment].iloc[0])
            closest_points1[shortest_segment] = find_closest_point(event.xdata, event.ydata, lines_df.loc[lines_df['index'] == shortest_segment].iloc[0])
            
        closest_x, closest_y = closest_points[shortest_segment]
        point, = plt.plot(closest_x, closest_y, 'o', color='black', markersize=5)
        plotted_points.append(point)

        print(closest_points1)
        plt.draw()

def save_closest_points(path, x_click, y_click):
    for segment in path:
        if segment not in closest_points:
            row = lines_df.loc[lines_df['index'] == segment].iloc[0] if segment in lines_df['index'].values else arcs_df.loc[arcs_df['index'] == segment].iloc[0]
            closest_points[segment] = find_closest_point(x_click, y_click, row)
    plot_dotted_path()

def plot_dotted_path():
    spacing = 0.15  # Set spacing between points (adjust as needed)
    for segment, point in closest_points.items():
        row = lines_df.loc[lines_df['index'] == segment].iloc[0] if segment in lines_df['index'].values else arcs_df.loc[arcs_df['index'] == segment].iloc[0]
        if segment in lines_df['index'].values:
            # Generate evenly spaced points for a line
            x_start, y_start = row['x'], row['y']
            x_end, y_end = calculate_endpoint(row)
            line_length = calculate_line_length(row)
            num_points = int(line_length / spacing)
            x_points = -1*np.linspace(x_start, x_end, num_points)
            y_points = -1*np.linspace(y_start, y_end, num_points)
            for x, y in zip(x_points, y_points):
                point, = plt.plot(x, y, 'o', color='purple', markersize=3)
                plotted_points.append(point)
        else:
            theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
            if row['Rotation'] == 'CW':
                theta = np.flip(theta)
            arc_length = calculate_arc_length(row)
            num_points = int(arc_length / spacing)
            theta_points = np.linspace(row['angleStart'], row['angleEnd'], num_points)
            x_arc = row['x'] + row['radius'] * np.cos(theta_points)
            y_arc = row['y'] + row['radius'] * np.sin(theta_points)
            for x, y in zip(-x_arc, -y_arc):
                point, = plt.plot(x, y, 'o', color='purple', markersize=3)
                plotted_points.append(point)

    plt.draw()

plot_all()

# Connect the click event and key press event
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Show the plot
plt.show()
