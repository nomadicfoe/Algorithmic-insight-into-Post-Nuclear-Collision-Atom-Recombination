import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KDTree

def read_coordinates_from_file(filename):
    coordinates = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.split()
            initial_pos = tuple(map(float, parts[1:4]))
            final_pos = tuple(map(float, parts[4:7]))
            coordinates.append((initial_pos, final_pos))
    return coordinates

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def find_nearest_final_positions(initial_positions, final_positions, distance_threshold):
    num_initial = len(initial_positions)
    num_final = len(final_positions)

    # Build the kd-tree from final positions
    final_positions_arr = np.array(final_positions)
    kd_tree = KDTree(final_positions_arr)

    # Create a cost matrix with distances between all initial and final positions
    cost_matrix = np.zeros((num_initial, num_final))
    for i in range(num_initial):
        for j in range(num_final):
            cost_matrix[i, j] = calculate_distance(initial_positions[i], final_positions[j])

    # Use the Hungarian algorithm to find the optimal matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a dictionary to store the nearest final positions for each initial position
    nearest_final_positions = {}
    for row_idx, col_idx in zip(row_indices, col_indices):
        initial_pos = initial_positions[row_idx]
        final_pos = final_positions[col_idx]
        distance = cost_matrix[row_idx, col_idx]
        if distance <= distance_threshold:
            nearest_final_positions[initial_pos] = final_pos

    return nearest_final_positions

# Read coordinates from file
filename = "C:/Users/suman/OneDrive/Desktop/BARC project/kd_brute/Pos3.dispxyz"
coordinates = read_coordinates_from_file(filename)

# Extract initial and final positions from the coordinates
initial_positions, final_positions = zip(*coordinates)

# Example usage
threshold = 8.0 # change as per requirements

nearest_final_positions = find_nearest_final_positions(initial_positions, final_positions, threshold)

for initial_pos, final_pos in nearest_final_positions.items():
    print("Initial position:", initial_pos, "Nearest final position:", final_pos)
