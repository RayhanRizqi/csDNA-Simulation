import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Function to simulate one robot journey through the maze
def simulate_robot(grid):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    start_position = (2, 0)
    position = start_position
    visited_counts = np.zeros_like(grid)
    path_length = 0
    
    while grid[position[0]][position[1]] != 2:  # Continue until goal is reached
        visited_counts[position[0], position[1]] += 1
        valid_moves = []
        for dx, dy in directions:
            nx, ny = position[0] + dx, position[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 3:
                valid_moves.append((nx, ny))
        if not valid_moves:
            break  # No valid moves
        position = random.choice(valid_moves)
        path_length += 1
    return visited_counts, path_length

# Define the maze grid
grid = np.array([
    [3, 0, 0, 3, 0, 0, 0, 3, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2]
])

# Run simulations
total_visits = np.zeros_like(grid)
path_lengths = []

for _ in range(100):
    visits, length = simulate_robot(grid)
    total_visits += visits
    path_lengths.append(length)

# Heat map of visits
plt.figure(figsize=(10, 5))
ax = plt.gca()
cax = ax.matshow(total_visits, cmap='hot', interpolation='nearest')
for (i, j), val in np.ndenumerate(total_visits):
    ax.text(j, i, f"{val}", ha='center', va='center', color='white')
plt.colorbar(cax)
plt.title('Heat Map of Robot Visits')
plt.show()

# Density plot of path lengths
path_lengths = np.array(path_lengths)
x = np.linspace(min(path_lengths), max(path_lengths), 300)
kde = gaussian_kde(path_lengths)
plt.figure(figsize=(10, 5))
plt.plot(x, kde(x), color='blue')
plt.title('Density Plot of Path Lengths to Goal')
plt.xlabel('Path Length')
plt.ylabel('Density')

# Customize x-axis for more granular detail
plt.xticks(np.arange(min(path_lengths), max(path_lengths) + 1, 100))  # Adjust this value to change interval

plt.show()
