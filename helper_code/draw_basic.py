import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV files into pandas dataframes
lines_df = pd.read_csv('Components/NewLines.csv')
arcs_df = pd.read_csv('Components/NewArcs.csv')

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

# Function to plot a line
def plot_line(row):
    x_start, y_start = row['x'], row['y']
    x_end, y_end = calculate_endpoint(row)
    
    # Plot the line
    plt.plot([-x_start, -x_end], [-y_start, -y_end], marker='o', label=row['index'], color='blue')

# Function to plot an arc
def plot_arc(row):
    theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
    if row['Rotation'] == 'CW':
        theta = np.flip(theta)
    
    x_arc = row['x'] + row['radius'] * np.cos(theta)
    y_arc = row['y'] + row['radius'] * np.sin(theta)
    
    # Plot the arc
    plt.plot(-x_arc, -y_arc, label=row['index'], color='red')

# Plot lines and arcs
plt.figure(figsize=(10, 10))

# Plot all lines
for _, row in lines_df.iterrows():
    plot_line(row)

# Plot all arcs
for _, row in arcs_df.iterrows():
    plot_arc(row)

# Adding labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lines and Arcs Plot from CSV Data')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()
