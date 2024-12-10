import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lines_df = pd.read_csv('Components/NewLines.csv')

def calculate_endpoint(row):
    if row['direction'] == '+X':
        return row['x'] - row['length'], row['y']
    elif row['direction'] == '-X':
        return row['x'] + row['length'], row['y']
    elif row['direction'] == '+Y':
        return row['x'], row['y'] - row['length']
    elif row['direction'] == '-Y':
        return row['x'], row['y'] + row['length']


plt.figure(figsize=(10, 10))

for _, row in lines_df.iterrows():
    x_start, y_start = row['x'], row['y']
    x_end, y_end = calculate_endpoint(row)
    
    plt.plot([-x_start, -x_end], [-y_start, -y_end], marker='o', label=row['index'])

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lines Plot from CSV Data')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
