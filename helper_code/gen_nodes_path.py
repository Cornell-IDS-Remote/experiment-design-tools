import pandas as pd
import numpy as np

processed_lines_df = pd.read_csv('Processed_Lines.csv')
processed_arcs_df = pd.read_csv('Processed_Arcs.csv')

def normalize(vector):
    """Normalize a 2D vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm

def are_tangents_opposite(tangent1, tangent2, tolerance=0.1):
    """
    Check if two tangents are nearly 180 degrees apart.
    Two vectors are opposite if their dot product is close to -1.
    """
    dot_product = np.dot(tangent1, tangent2)
    return np.isclose(dot_product, -1, atol=tolerance)

def find_connections(current_row, all_rows):
    """
    Find arcs/lines that are connected to the current one, based only on proximity of endpoints.
    """
    connections = []
    
    current_endpoints = {
        'endpoint_1': np.array([current_row['starting_x'], current_row['starting_y']]),
        'endpoint_2': np.array([current_row['ending_x'], current_row['ending_y']])
    }
    
    for _, other_row in all_rows.iterrows():
        if current_row['index'] == other_row['index']:
            continue  # Skip the same path
        
        other_endpoints = {
            'endpoint_1': np.array([other_row['starting_x'], other_row['starting_y']]),
            'endpoint_2': np.array([other_row['ending_x'], other_row['ending_y']])
        }
        
        # Check if any of the endpoints are close
        for curr_endpoint in current_endpoints.values():
            for other_endpoint in other_endpoints.values():
                if np.allclose(curr_endpoint, other_endpoint, atol=0.1):
                    connections.append(other_row['index'])
    
    return connections

def find_paths(current_row, all_rows, connections):
    """
    Find valid paths where the connection transitions to another line/arc, based on endpoint proximity
    and opposite tangents.
    """
    paths = {}
    
    current_endpoints = {
        'endpoint_1': np.array([current_row['starting_x'], current_row['starting_y']]),
        'endpoint_2': np.array([current_row['ending_x'], current_row['ending_y']])
    }
    current_tangents = {
        'tangent_1': normalize(np.array([current_row['tangent1_x'], current_row['tangent1_y']])),
        'tangent_2': normalize(np.array([current_row['tangent2_x'], current_row['tangent2_y']]))
    }

    # Iterate over the current node's connections
    for conn_id in connections:
        conn_row = all_rows[all_rows['index'] == conn_id].iloc[0]
        conn_endpoints = {
            'endpoint_1': np.array([conn_row['starting_x'], conn_row['starting_y']]),
            'endpoint_2': np.array([conn_row['ending_x'], conn_row['ending_y']])
        }
        conn_tangents = {
            'tangent_1': normalize(np.array([conn_row['tangent1_x'], conn_row['tangent1_y']])),
            'tangent_2': normalize(np.array([conn_row['tangent2_x'], conn_row['tangent2_y']]))
        }

        for curr_end_key, curr_endpoint in current_endpoints.items():
            for conn_end_key, conn_endpoint in conn_endpoints.items():
                if np.allclose(curr_endpoint, conn_endpoint, atol=0.2):
                    # Check if their tangents are opposite
                    curr_tangent_key = 'tangent_1' if curr_end_key == 'endpoint_1' else 'tangent_2'
                    conn_tangent_key = 'tangent_1' if conn_end_key == 'endpoint_1' else 'tangent_2'
                    
                    if are_tangents_opposite(current_tangents[curr_tangent_key], conn_tangents[conn_tangent_key]):
                        if curr_end_key not in paths:
                            paths[curr_end_key] = []
                        paths[curr_end_key].append(conn_id)


    segments = {}
    if 'endpoint_1' in paths and 'endpoint_2' in paths:
        list1 = paths['endpoint_1']
        list2 = paths['endpoint_2']
        for l1 in list1:
            for i, l2 in enumerate(list2):
                if i == 0:
                    segments[l1] = [l2]
                else:
                    segments[l1].append(l2)
        for l2 in list2:
            for i, l1 in enumerate(list1):
                if i == 0:
                    segments[l2] = [l1]
                else:
                    segments[l2].append(l1)
    
    return segments

class Node:
    def __init__(self, id, all_rows):
        self.id = id
        self.connections = self.find_connections(all_rows)
        self.paths = self.find_paths(all_rows)

    def find_connections(self, all_rows):
        """Find all connections based on endpoints proximity."""
        current_row = all_rows[all_rows['index'] == self.id].iloc[0]
        return find_connections(current_row, all_rows)

    def find_paths(self, all_rows):
        """Find valid paths where incoming and outgoing tangents are opposite."""
        current_row = all_rows[all_rows['index'] == self.id].iloc[0]
        return find_paths(current_row, all_rows, self.connections)

def instantiate_nodes(ids, all_rows):
    """
    Create Node instances for the given line/arc IDs and return a list of them.
    """
    nodes = []
    node_data = []
    for id in ids:
        node = Node(id, all_rows)
        nodes.append(node)
        node_data.append({'id': node.id, 'paths': node.paths})
    return node_data

all_data_df = pd.concat([processed_lines_df, processed_arcs_df])

all_arc_indexes = processed_arcs_df['index'].tolist()  # Get all arc IDs
all_line_indexes = processed_lines_df['index'].tolist()  # Get all line IDs

all_indexes = all_arc_indexes + all_line_indexes

nodes_data = instantiate_nodes(all_indexes, all_data_df)

nodes_df = pd.DataFrame(nodes_data)

nodes_df.to_csv('nodes_paths.csv', index=False)

print("Node paths have been saved to 'nodes_paths.csv'")
