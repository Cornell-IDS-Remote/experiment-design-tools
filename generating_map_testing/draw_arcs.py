import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arcs_df = pd.read_csv('Components/NewArcs.csv')

def plot_arc(row):
    theta = np.linspace(row['angleStart'], row['angleEnd'], 100)
    if row['Rotation'] == 'CW':
        theta = np.flip(theta)
    
    x_arc = row['x'] + row['radius'] * np.cos(theta)
    y_arc = row['y'] + row['radius'] * np.sin(theta)
    
    plt.plot(-x_arc, -y_arc, label=row['index'])

plt.figure(figsize=(10, 10))

for _, row in arcs_df.iterrows():
    plot_arc(row)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Arcs Plot from CSV Data')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
