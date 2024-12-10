import pandas as pd
import numpy as np

lines_df = pd.read_csv('Components/NewLines.csv')
arcs_df = pd.read_csv('Components/NewArcs.csv')

def normalize(vector):
    """Normalize a 2D vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm

class Line:
    def __init__(self, x, y, direction, length):
        self.endpoint_1 = np.array([x, y])
        self.endpoint_2 = np.array(self.calculate_endpoint(x, y, direction, length))
        self.tangent_end_point_1 = self.calculate_tangent(True, direction)
        self.tangent_end_point_2 = self.calculate_tangent(False, direction)

    def calculate_endpoint(self, x, y, direction, length):
        if direction == '+X':
            return x - length, y
        elif direction == '-X':
            return x + length, y
        elif direction == '+Y':
            return x, y - length
        elif direction == '-Y':
            return x, y + length

    def calculate_tangent(self, first, direction):
        k = 1
        if not first:
            k = -1

        if direction == '+X':
            return k * np.array([1, 0])
        elif direction == '-X':
            return k * np.array([-1, 0])
        elif direction == '+Y':
            return k * np.array([0, 1])
        elif direction == '-Y':
            return k * np.array([0, -1])


class Arc:
    def __init__(self, x, y, radius, angle_start, angle_end, rotation):
        self.endpoint_1 = self.calculate_arc_endpoint(x, y, radius, angle_start)
        self.endpoint_2 = self.calculate_arc_endpoint(x, y, radius, angle_end)
        self.tangent_end_point_1 = -1*self.calculate_tangent(self.endpoint_1[0], self.endpoint_1[1], angle_start)
        self.tangent_end_point_2 = self.calculate_tangent(self.endpoint_2[0], self.endpoint_2[1], angle_end)
        self.rotation = rotation

    def calculate_arc_endpoint(self, x, y, radius, angle):
        return np.array([x + radius * np.cos(angle), y + radius * np.sin(angle)])

    def calculate_tangent(self, x, y, angle):
        # Derive the tangent vector from the angle
        return np.array([-np.sin(angle), np.cos(angle)])  # Perpendicular to the radius

processed_lines = []
processed_arcs = []

for _, row in lines_df.iterrows():
    line = Line(row['x'], row['y'], row['direction'], row['length'])
    tangent_1_normalized = normalize(line.tangent_end_point_1)
    tangent_2_normalized = normalize(line.tangent_end_point_2)
    processed_lines.append({
        **row,
        'starting_x': line.endpoint_1[0],
        'starting_y': line.endpoint_1[1],
        'ending_x': line.endpoint_2[0],
        'ending_y': line.endpoint_2[1],
        'tangent1_x': tangent_1_normalized[0],
        'tangent1_y': tangent_1_normalized[1],
        'tangent2_x': tangent_2_normalized[0],
        'tangent2_y': tangent_2_normalized[1]
    })

for _, row in arcs_df.iterrows():
    arc = Arc(row['x'], row['y'], row['radius'], row['angleStart'], row['angleEnd'], row['Rotation'])
    tangent_1_normalized = normalize(arc.tangent_end_point_1)
    tangent_2_normalized = normalize(arc.tangent_end_point_2)
    processed_arcs.append({
        **row,
        'starting_x': arc.endpoint_1[0],
        'starting_y': arc.endpoint_1[1],
        'ending_x': arc.endpoint_2[0],
        'ending_y': arc.endpoint_2[1],
        'tangent1_x': tangent_1_normalized[0],
        'tangent1_y': tangent_1_normalized[1],
        'tangent2_x': tangent_2_normalized[0],
        'tangent2_y': tangent_2_normalized[1]
    })

processed_lines_df = pd.DataFrame(processed_lines).round(3)
processed_arcs_df = pd.DataFrame(processed_arcs).round(3)

# Output the data to new CSV files
processed_lines_df.to_csv('Processed_Lines.csv', index=False)
processed_arcs_df.to_csv('Processed_Arcs.csv', index=False)

print("Processing complete. New CSV files saved.")
