import numpy as np
import random
import tkinter as tk
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Function to simulate one robot journey through the maze
def simulate_robot(grid):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    start_position = (2, 0)
    position = start_position
    visited_counts = np.zeros_like(grid)
    path = [start_position]
    path_length = 0
    
    while True:
        visited_counts[position[0], position[1]] += 1
        if grid[position[0]][position[1]] == 2:
            break  # Stop moving when goal is reached
        valid_moves = []
        for dx, dy in directions:
            nx, ny = position[0] + dx, position[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 3:
                valid_moves.append((nx, ny))
        if not valid_moves:
            break  # No valid moves
        position = random.choice(valid_moves)
        path.append(position)
        path_length += 1
    return visited_counts, path_length, path

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
paths = []

for _ in range(100):
    visits, length, path = simulate_robot(grid)
    total_visits += visits
    path_lengths.append(length)
    paths.append(path)

# Heat map of visits
plt.figure(figsize=(10, 5))
ax = plt.gca()
cax = ax.matshow(total_visits, cmap='hot', interpolation='nearest')
for (i, j), val in np.ndenumerate(total_visits):
    ax.text(j, i, f"{val}", ha='center', va='center', color='white')
plt.colorbar(cax)
plt.title('Heat Map of Robot Visits Including Goal Cell Visits')
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
plt.xticks(np.arange(min(path_lengths), max(path_lengths) + 1, 50))
plt.show()

# Find the shortest path
shortest_path_index = np.argmin(path_lengths)
shortest_path = paths[shortest_path_index]

# Define the colors for fading effect
trail_colors = ["#ff0000", "#ff4d4d", "#ff9999", "#ffc2c2", "#ffe6e6", "white"]

# Create the main window for the Tkinter GUI
root = tk.Tk()
root.title("Shortest Path Simulation with Trail Effect")

# Create a frame to hold the grid
frame = tk.Frame(root, bg="black")
frame.pack(padx=5, pady=5)

# Define colors for the cells
colors = {0: "white", 1: "green", 2: "red", 3: "black"}

# Create labels for each cell in the grid and a dictionary to track cell colors
labels = [[None for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
cell_colors = [[colors[grid[i, j]] for j in range(grid.shape[1])] for i in range(grid.shape[0])]

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        label = tk.Label(frame, bg=cell_colors[i][j], width=10, height=5, relief="solid", borderwidth=1)
        label.grid(row=i, column=j, sticky="nsew")
        labels[i][j] = label

# Step counter label
step_counter_label = tk.Label(root, text="Step: 0", font=('Helvetica', 14))
step_counter_label.pack()

def update_cell_colors():
    """Update colors for cells in the fading trail."""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if cell_colors[i][j] in trail_colors:
                current_color_index = trail_colors.index(cell_colors[i][j])
                if current_color_index < len(trail_colors) - 1:
                    cell_colors[i][j] = trail_colors[current_color_index + 1]
                    labels[i][j].config(bg=cell_colors[i][j])

def visualize_path(path):
    previous_position = None
    for step, position in enumerate(path):
        if previous_position is not None:
            # Mark the previous position with the starting color of the trail
            cell_colors[previous_position[0]][previous_position[1]] = trail_colors[0]
            labels[previous_position[0]][previous_position[1]].config(bg=cell_colors[previous_position[0]][previous_position[1]])

        x, y = position
        # Update step counter
        step_counter_label.config(text=f"Step: {step + 1}")
        # Update the cell colors for fading effect
        update_cell_colors()
        # Mark the current position with blue
        labels[x][y].config(bg='blue')
        root.update()
        time.sleep(0.5)  # Slow down the simulation

        if grid[x][y] == 2:  # Stop at the goal
            break

        # Update the previous position for the next step
        previous_position = position

# Button to start the path visualization
start_button = tk.Button(root, text="Start Simulation", command=lambda: visualize_path(shortest_path))
start_button.pack()

# Start the Tkinter main loop
root.mainloop()
