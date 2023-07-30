import numpy as np
from scipy.optimize import linear_sum_assignment

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

def find_nearest_final_positions(initial_positions, final_positions, distance_threshold):
    num_initial = len(initial_positions)
    num_final = len(final_positions)

    # Calculate the distances between initial and final positions
    distances = np.zeros((num_initial, num_final))
    for i in range(num_initial):
        for j in range(num_final):
            distances[i, j] = np.linalg.norm(np.array(initial_positions[i]) - np.array(final_positions[j]))

    # Solve the assignment problem to find the minimum weight matching
    row_ind, col_ind = linear_sum_assignment(distances)

    # Create a dictionary to store the nearest final positions within the distance threshold for each initial position
    nearest_final_positions = {}
    for i in range(num_initial):
        initial_pos = initial_positions[i]
        final_pos = final_positions[col_ind[i]]
        distance = distances[row_ind[i], col_ind[i]]
        if distance <= distance_threshold:
            nearest_final_positions[initial_pos] = final_pos

    return nearest_final_positions

# Read coordinates from file
filename = "C:/Users/suman/OneDrive/Desktop/BARC project/kd_brute/Pos2.dispxyz"
coordinates = read_coordinates_from_file(filename)

# Extract initial and final positions from the coordinates
initial_positions, final_positions = zip(*coordinates)

# Example usage
threshold = 8.0 # change as per requirements

nearest_final_positions = find_nearest_final_positions(initial_positions, final_positions, threshold)

for initial_pos, final_pos in nearest_final_positions.items():
    print("Initial position:", initial_pos, "Nearest final position:", final_pos)
